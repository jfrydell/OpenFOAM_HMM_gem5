/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2017-2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "GaussSeidelSmoother.H"
#include "PrecisionAdaptor.H"


#ifdef USE_ROCTX
#include <roctracer/roctx.h>
#endif

#ifdef USE_OMP
#include <omp.h>
  #ifndef OMP_UNIFIED_MEMORY_REQUIRED
  #pragma omp requires unified_shared_memory
  #define OMP_UNIFIED_MEMORY_REQUIRED
  #endif
#endif

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(GaussSeidelSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<GaussSeidelSmoother>
        addGaussSeidelSmootherSymMatrixConstructorToTable_;

    lduMatrix::smoother::addasymMatrixConstructorToTable<GaussSeidelSmoother>
        addGaussSeidelSmootherAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::GaussSeidelSmoother::GaussSeidelSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::GaussSeidelSmoother::smooth
(
    const word& fieldName_,
    solveScalarField& psi,
    const lduMatrix& matrix_,
    const solveScalarField& source,
    const FieldField<Field, scalar>& interfaceBouCoeffs_,
    const lduInterfaceFieldPtrsList& interfaces_,
    const direction cmpt,
    const label nSweeps
)
{
    solveScalar* __restrict__ psiPtr = psi.begin();

    const label nCells = psi.size();

    solveScalarField bPrime(nCells);
    solveScalar* __restrict__ bPrimePtr = bPrime.begin();

    const scalar* const __restrict__ diagPtr = matrix_.diag().begin();
    const scalar* const __restrict__ upperPtr =
        matrix_.upper().begin();
    const scalar* const __restrict__ lowerPtr =
        matrix_.lower().begin();

    const label* const __restrict__ uPtr =
        matrix_.lduAddr().upperAddr().begin();

    const label* const __restrict__ lPtr =
        matrix_.lduAddr().lowerAddr().begin();

    const label* const __restrict__ ownStartPtr =
        matrix_.lduAddr().ownerStartAddr().begin();


    // Parallel boundary initialisation.  The parallel boundary is treated
    // as an effective jacobi interface in the boundary.
    // Note: there is a change of sign in the coupled
    // interface update.  The reason for this is that the
    // internal coefficients are all located at the l.h.s. of
    // the matrix whereas the "implicit" coefficients on the
    // coupled boundaries are all created as if the
    // coefficient contribution is of a source-kind (i.e. they
    // have a sign as if they are on the r.h.s. of the matrix.
    // To compensate for this, it is necessary to turn the
    // sign of the contribution.

    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        bPrime = source;

        const label startRequest = Pstream::nRequests();

        matrix_.initMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt
        );

        matrix_.updateMatrixInterfaces
        (
            false,
            interfaceBouCoeffs_,
            interfaces_,
            psi,
            bPrime,
            cmpt,
            startRequest
        );

        solveScalar psii;
        label fStart;
        label fEnd = ownStartPtr[0];

#if 1
        //temporary field
        solveScalarField Z(psi.size()); //temporary array 
        solveScalarField R(psi.size()); //residual

        solveScalar* __restrict__ rhs_ptr = bPrime.begin();
	solveScalar* __restrict__ r_ptr = R.begin();
        solveScalar* __restrict__ u_ptr = psi.begin();
        solveScalar* __restrict__ z_ptr = Z.begin();

	const label nFaces = matrix_.upper().size();

	const label USE_ZERO_ORDER = 0;

	//use relax_weight = 1.0;
        // 0) r = relax_weight * (RHS - A * u)
	
        #pragma omp target teams distribute parallel for if(nCells > 10000)
        for (label celli=0; celli<nCells; ++celli)
        {
          r_ptr[celli] = diagPtr[celli]*u_ptr[celli];
        }


        #pragma omp target teams distribute parallel for if(nFaces > 10000) thread_limit(256)
        for (label face=0; face<nFaces; face+=2)
        {
            const label nf = (nFaces-face) > 1 ? 2 : 1;
            #pragma unroll 2
            for ( label i = 0; i < nf; ++i){
              const label l_val = lPtr[face+i];
              const label u_val = uPtr[face+i];
              #pragma omp atomic
              r_ptr[u_val] += lowerPtr[face+i]*u_ptr[l_val];
              #pragma omp atomic
              r_ptr[l_val] += upperPtr[face+i]*u_ptr[u_val];
            }
        }



	if (1 == USE_ZERO_ORDER){

	   scalar relax_weight = 0.8;

            // 1) z = r/D, u = u + relax_weight * z
           #pragma omp target teams distribute parallel for if(nCells > 10000)
           for (label celli=0; celli<nCells; celli++)
           {
             scalar r = rhs_ptr[celli] - r_ptr[celli];
             u_ptr[celli] += relax_weight * r / diagPtr[celli];
           }
	}
        else{

        // 1) z = r/D, u = u + z
	#pragma omp target teams distribute parallel for if(nCells > 10000)
        for (label celli=0; celli<nCells; celli++)
	{
          scalar r = rhs_ptr[celli] - r_ptr[celli];
          z_ptr[celli] = r / diagPtr[celli];
	  u_ptr[celli] += z_ptr[celli]; 
	}

        scalar multiplier = -1.0;

	for (label sweepID = 0; sweepID < 1; sweepID++)
        {
            // 2) r = U * z

	    #pragma omp target teams distribute parallel for if(nCells > 10000)
            for (label celli=0; celli<nCells; celli++)
	    {
              fStart = ownStartPtr[celli];
              fEnd   = ownStartPtr[celli + 1];

	      scalar tmp = 0.0;
              #pragma unroll 4
              for (label facei=fStart; facei<fEnd; facei++)
              {
                  tmp +=  upperPtr[facei]*z_ptr[uPtr[facei]];
              }
              r_ptr[celli] = tmp;
	    }

	    // 3) z = r/D, u = u + m * z
	    #pragma omp target teams distribute parallel for if(nCells > 10000)
	    for (label celli=0; celli<nCells; celli++)
            {
              z_ptr[celli] = r_ptr[celli] / diagPtr[celli];
	      u_ptr[celli] += multiplier * z_ptr[celli];
	    }
	    multiplier *= -1.0;
   	}
	}

#else

        for (label celli=0; celli<nCells; celli++)
        {
            // Start and end of this row
            fStart = fEnd;
            fEnd = ownStartPtr[celli + 1];

            // Get the accumulated neighbour side
            psii = bPrimePtr[celli];

            // Accumulate the owner product side
            for (label facei=fStart; facei<fEnd; facei++)
            {
                psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
            }

            // Finish psi for this cell
            psii /= diagPtr[celli];

            // Distribute the neighbour side using psi for this cell
            for (label facei=fStart; facei<fEnd; facei++)
            {
                bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
            }

            psiPtr[celli] = psii;
        }
#endif

    }
}


void Foam::GaussSeidelSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_,
        psi,
        matrix_,
        ConstPrecisionAdaptor<solveScalar, scalar>(source),
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


void Foam::GaussSeidelSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_,
        psi,
        matrix_,
        source,
        interfaceBouCoeffs_,
        interfaces_,
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
