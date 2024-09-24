/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

#include "fvPatch.H"

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


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::fvPatch::patchInternalField
(
    const UList<Type>& f
) const
{
    return patchInternalField(f, this->faceCells());
}


template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::fvPatch::patchInternalField
(
    const UList<Type>& f,
    const labelUList& faceCells
) const
{
    #ifdef USE_ROCTX
    roctxRangePush("fvPatch::patchInternalField-1");
    #endif

    auto tpif = tmp<Field<Type>>::New(size());
    auto& pif = tpif.ref();

    //forAll(pif, facei)
    const label loop_len = pif.size();
    #pragma omp target teams distribute parallel for thread_limit(128) if(loop_len > 3000)
    for (label facei = 0; facei < loop_len; ++facei)
    {
        pif[facei] = f[faceCells[facei]];
    }

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


    return tpif;
}


template<class Type>
void Foam::fvPatch::patchInternalField
(
    const UList<Type>& f,
    Field<Type>& pif
) const
{

    #ifdef USE_ROCTX
    roctxRangePush("fvPatch::patchInternalField-2");
    #endif
    pif.resize(size());

    const labelUList& faceCells = this->faceCells();

    //forAll(pif, facei)
    const label loop_len = pif.size();
    #pragma omp target teams distribute parallel for thread_limit(128) if(loop_len > 3000)
    for (label facei = 0; facei < loop_len; ++facei)
    {
        pif[facei] = f[faceCells[facei]];
    }

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif
}


template<class GeometricField, class Type>
const typename GeometricField::Patch& Foam::fvPatch::patchField
(
    const GeometricField& gf
) const
{
    return gf.boundaryField()[index()];
}


// ************************************************************************* //
