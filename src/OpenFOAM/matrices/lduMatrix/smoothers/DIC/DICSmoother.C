/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

#include "DICSmoother.H"
#include "DICPreconditioner.H"
#include "PrecisionAdaptor.H"
#include <algorithm>

#ifndef OMP_UNIFIED_MEMORY_REQUIRED
#pragma omp requires unified_shared_memory
#define OMP_UNIFIED_MEMORY_REQUIRED
#endif

#ifdef USE_ROCTX
#include <roctracer/roctx.h>
#endif



// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DICSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<DICSmoother>
        addDICSmootherSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DICSmoother::DICSmoother
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
    ),
    rD_(matrix_.diag().size())
{
    const scalarField& diag = matrix_.diag();
    std::copy(diag.begin(), diag.end(), rD_.begin());

    DICPreconditioner::calcReciprocalD(rD_, matrix_);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::DICSmoother::smooth
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    const solveScalar* const __restrict__ rDPtr = rD_.begin();
    const scalar* const __restrict__ upperPtr = matrix_.upper().begin();
    const label* const __restrict__ uPtr =
        matrix_.lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr =
        matrix_.lduAddr().lowerAddr().begin();



    #ifdef USE_ROCTX
    roctxRangePush("DICSmoother::smooth");
    #endif


    // Temporary storage for the residual
    solveScalarField rA(rD_.size());
    solveScalar* __restrict__ rAPtr = rA.begin();

    #ifdef DICSmoother_PARALLEL
    solveScalarField rA_temp(rA.size());
    solveScalar* __restrict__ rA_temp_Ptr = rA_temp.begin();
    #endif


    for (label sweep=0; sweep<nSweeps; sweep++)
    {
        matrix_.residual
        (
            rA,
            psi,
            source,
            interfaceBouCoeffs_,
            interfaces_,
            cmpt
        );
#ifdef DICSmoother_PARALLEL

        //forAll(rA, i)
        const label loop_len = rA.size();
        #pragma omp target teams distribute parallel for if(loop_len > 5000)
	for (label i = 0; i < loop_len; ++i)
        {
            rAPtr[i] *= rDPtr[i];
	    rA_temp_Ptr[i] = rAPtr[i];
        }


        const label nFaces = matrix_.upper().size();
	#pragma omp target teams distribute parallel for if(nFaces > 5000)
        for (label facei=0; facei<nFaces; facei++)
        {
            const label u = uPtr[facei];
            rA_temp_Ptr[u] -= rDPtr[u]*upperPtr[facei]*rAPtr[lPtr[facei]];
        }

        const label nFacesM1 = nFaces - 1;
	#pragma omp target teams distribute parallel for if(nFaces > 5000)
        for (label facei=nFacesM1; facei>=0; facei--)
        {
            const label l = lPtr[facei];
            rAPtr[l] -= rDPtr[l]*upperPtr[facei]*rA_temp_Ptr[uPtr[facei]];
        }

#else

        forAll(rA, i)
        {
            rA[i] *= rD_[i];
        }

        const label nFaces = matrix_.upper().size();
        for (label facei=0; facei<nFaces; facei++)
        {
            const label u = uPtr[facei];
            rAPtr[u] -= rDPtr[u]*upperPtr[facei]*rAPtr[lPtr[facei]];
        }

        const label nFacesM1 = nFaces - 1;
        for (label facei=nFacesM1; facei>=0; facei--)
        {
            const label l = lPtr[facei];
            rAPtr[l] -= rDPtr[l]*upperPtr[facei]*rAPtr[uPtr[facei]];
        }
#endif

        psi += rA;
    }



    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


}


void Foam::DICSmoother::scalarSmooth
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        psi,
        ConstPrecisionAdaptor<scalar, solveScalar>(source),
        cmpt,
        nSweeps
    );
}


// ************************************************************************* //
