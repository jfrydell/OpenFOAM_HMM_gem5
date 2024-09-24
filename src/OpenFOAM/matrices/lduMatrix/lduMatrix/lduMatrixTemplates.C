/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Description
    lduMatrix member H operations.

\*---------------------------------------------------------------------------*/
#ifdef USE_OMP
  #include <omp.h>
  #ifndef OMP_UNIFIED_MEMORY_REQUIRED
  #pragma omp requires unified_shared_memory
  #define OMP_UNIFIED_MEMORY_REQUIRED
  #endif
#endif

#include "lduMatrix.H"

#include "AtomicAccumulator.H"
#include "macros.H"


// #define USE_OPT_LDU

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::lduMatrix::H(const Field<Type>& psi) const
{
    tmp<Field<Type>> tHpsi
    (
        new Field<Type>(lduAddr().size(), Zero)
    );

    if (lowerPtr_ || upperPtr_)
    {
        Field<Type> & Hpsi = tHpsi.ref();

        Type* __restrict__ HpsiPtr = Hpsi.begin();

        const Type* __restrict__ psiPtr = psi.begin();

        const label* __restrict__ uPtr = lduAddr().upperAddr().begin();
        const label* __restrict__ lPtr = lduAddr().lowerAddr().begin();

        const scalar* __restrict__ lowerPtr = lower().begin();
        const scalar* __restrict__ upperPtr = upper().begin();

        const label nFaces = upper().size();
       
        #ifdef USE_OPT_LDU
        static label *repetition_count_ofsetts = NULL;

        static label counter=0;
        if (repetition_count_ofsetts == NULL){

          label *repetition_count = new (std::align_val_t(256)) label[nFaces];
          label face = 0;
          while (face < nFaces){
            const label l_val = lPtr[face] ;
            label local_counter=0;
            while (counter<nFaces){
                 if (l_val == lPtr[face+local_counter])
                    local_counter++;
                 else
                    break;
            }
            repetition_count[counter] = local_counter;
            face += local_counter;
            counter++;
         }
         repetition_count_ofsetts = new (std::align_val_t(256)) label[counter+1];
         repetition_count_ofsetts[0] = 0;
         for (label i=1; i <= counter; i++)
           repetition_count_ofsetts[i] = repetition_count_ofsetts[i-1]+repetition_count[i-1];

         delete[] repetition_count;
       }
       #endif



    //    double t1 = omp_get_wtime();	


        #ifdef USE_OPT_LDU


         #pragma omp target teams distribute parallel for
         for (label i=0; i < counter; i++){
	    Type sum; //need to figure out how to properly initialize

	    //if constexpr ( std::is_same<Type,scalar>() ) sum = 0;
	    //else sum = Foam::vector<scalar>(0,0,0);

            const label face_start = repetition_count_ofsetts[i];
            const label face_stop = repetition_count_ofsetts[i+1];

            const label l_val = lPtr[face_start];
            const Type psiPtr_l_val = psiPtr[l_val];

            #pragma unroll 2
            for (label face = face_start; face < face_stop; ++face){

                const label u_val = uPtr[face];

		atomicAccumulator(HpsiPtr[u_val]) -= (lowerPtr[face+i]*psiPtr_l_val);
		sum                               +=  upperPtr[face+i]*psiPtr[u_val];
	    }
	    atomicAccumulator(HpsiPtr[l_val]) -= (sum);
         }
       
        #else

        #pragma omp target teams distribute parallel for thread_limit(256)  if(nFaces>10000) 
        for (label face=0; face<nFaces; face+=2)
        {
	    const label nf = (nFaces-face) > 1 ? 2 : 1;
            #pragma unroll 2
            for ( label i = 0; i < nf; ++i){
              const label l_val = lPtr[face+i] ;
              const label u_val = uPtr[face+i];
              atomicAccumulator(HpsiPtr[u_val]) -= (lowerPtr[face+i]*psiPtr[l_val]);
              atomicAccumulator(HpsiPtr[l_val]) -= (upperPtr[face+i]*psiPtr[u_val]);
	    }
        }

        #endif


//	double t2 = omp_get_wtime();
//	fprintf(stderr,"rank=%d: nFaces = %d, line=%d, loop time = %g\n",Pstream::myProcNo(), nFaces, t2-t1);
    }

    return tHpsi;
}

template<class Type>
Foam::tmp<Foam::Field<Type>>
Foam::lduMatrix::H(const tmp<Field<Type>>& tpsi) const
{
    tmp<Field<Type>> tHpsi(H(tpsi()));
    tpsi.clear();
    return tHpsi;
}


template<class Type>
Foam::tmp<Foam::Field<Type>>
Foam::lduMatrix::faceH(const Field<Type>& psi) const
{
    if (lowerPtr_ || upperPtr_)
    {
        const scalarField& Lower = const_cast<const lduMatrix&>(*this).lower();
        const scalarField& Upper = const_cast<const lduMatrix&>(*this).upper();

        const labelUList& l = lduAddr().lowerAddr();
        const labelUList& u = lduAddr().upperAddr();

        tmp<Field<Type>> tfaceHpsi(new Field<Type> (Lower.size()));
        Field<Type> & faceHpsi = tfaceHpsi.ref();

	label loop_len = l.size();
        #pragma omp target teams distribute parallel for if(loop_len>10000)
        for (label face=0; face<loop_len; face++)
        {
            faceHpsi[face] =
                Upper[face]*psi[u[face]]
              - Lower[face]*psi[l[face]];
        }

        return tfaceHpsi;
    }

    FatalErrorInFunction
        << "Cannot calculate faceH"
           " the matrix does not have any off-diagonal coefficients."
        << exit(FatalError);

    return nullptr;
}


template<class Type>
Foam::tmp<Foam::Field<Type>>
Foam::lduMatrix::faceH(const tmp<Field<Type>>& tpsi) const
{
    tmp<Field<Type>> tfaceHpsi(faceH(tpsi()));
    tpsi.clear();
    return tfaceHpsi;
}


// ************************************************************************* //
