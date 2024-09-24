/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2021 OpenCFD Ltd.
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

#include "cellMDLimitedGrad.H"
#include "gaussGrad.H"
#include "fvMesh.H"
#include "volMesh.H"
#include "surfaceMesh.H"
#include "volFields.H"
#include "fixedValueFvPatchFields.H"


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


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makeFvGradScheme(cellMDLimitedGrad)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<>
Foam::tmp<Foam::volVectorField>
Foam::fv::cellMDLimitedGrad<Foam::scalar>::calcGrad
(
    const volScalarField& vsf,
    const word& name
) const
{

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellMDLimitedGrad::calcGrad");
    #endif

    const fvMesh& mesh = vsf.mesh();

    tmp<volVectorField> tGrad = basicGradScheme_().calcGrad(vsf, name);

    if (k_ < SMALL)
    {
        return tGrad;
    }

    volVectorField& g = tGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const volVectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    scalarField maxVsf(vsf.primitiveField());
    scalarField minVsf(vsf.primitiveField());

    const label loop_len = owner.size();
    #pragma omp target teams distribute parallel for if(loop_len > 10000)
    for (label facei = 0; facei < loop_len; ++facei)
    //forAll(owner, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        const scalar vsfOwn = vsf[own];
        const scalar vsfNei = vsf[nei];
        #if 1
	  #pragma omp atomic compare
          maxVsf[own] = maxVsf[own] < vsfNei ? vsfNei : maxVsf[own];
	  #pragma omp atomic compare
          minVsf[own] = minVsf[own] > vsfNei ? vsfNei : minVsf[own];
          #pragma omp atomic compare
          maxVsf[nei] = maxVsf[nei] < vsfOwn ? vsfOwn : maxVsf[nei];
	  #pragma omp atomic compare
          minVsf[nei] = minVsf[nei] > vsfOwn ? vsfOwn : minVsf[nei];
        #else
          maxVsf[own] = max(maxVsf[own], vsfNei);
          minVsf[own] = min(minVsf[own], vsfNei);

          maxVsf[nei] = max(maxVsf[nei], vsfOwn);
          minVsf[nei] = min(minVsf[nei], vsfOwn);
        #endif
    }


    const volScalarField::Boundary& bsf = vsf.boundaryField();

    #ifdef USE_ROCTX
    roctxRangePush("cellMDLimitedGrad::calcGrad:loop2");
    #endif

    forAll(bsf, patchi)
    {
        const fvPatchScalarField& psf = bsf[patchi];

        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            const scalarField psfNei(psf.patchNeighbourField());

	    const label loop_len = pOwner.size();
	      #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 5000)
              for (label pFacei = 0; pFacei < loop_len; pFacei+=2)
            //  forAll(pOwner, pFacei)
              {
                const label nf = (loop_len-pFacei) > 1 ? 2 : 1;
                #pragma unroll 2
                for ( label i = 0; i < nf; ++i){
                  const label own = pOwner[pFacei+i];
                  const scalar vsfNei = psfNei[pFacei+i];

                  #if 1
		    #pragma omp atomic compare
                    maxVsf[own] = maxVsf[own] < vsfNei ? vsfNei : maxVsf[own] ;
	  	    #pragma omp atomic compare
		    minVsf[own] = minVsf[own] > vsfNei ? vsfNei : minVsf[own]; 
                  #else
                    maxVsf[own] = max(maxVsf[own], vsfNei);
                    minVsf[own] = min(minVsf[own], vsfNei);
                  #endif
		}
 	      }
        }
        else
        {
	    const label loop_len = pOwner.size();
            #pragma omp target teams distribute parallel for if(loop_len > 10000)
            for (label pFacei = 0; pFacei < loop_len; ++pFacei)
            //forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                const scalar vsfNei = psf[pFacei];
                #if 1
                  #pragma omp atomic compare
                  maxVsf[own] = maxVsf[own] < vsfNei ? vsfNei : maxVsf[own] ;
                  #pragma omp atomic compare
                  minVsf[own] = minVsf[own] > vsfNei ? vsfNei : minVsf[own];
                #else
                  maxVsf[own] = max(maxVsf[own], vsfNei);
                  minVsf[own] = min(minVsf[own], vsfNei);
                #endif
            }
        }
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


    maxVsf -= vsf;
    minVsf -= vsf;

    if (k_ < 1.0)
    {
        const scalarField maxMinVsf((1.0/k_ - 1.0)*(maxVsf - minVsf));
        maxVsf += maxMinVsf;
        minVsf -= maxMinVsf;

        //maxVsf *= 1.0/k_;
        //minVsf *= 1.0/k_;
    }
#if 0
    label g_len=g.size();
    int *test_unique_g = new int[g_len];
    for (label i=0; i < g_len; ++i) test_unique_g[i] = 0;

    for (label facei=0; facei < owner.size(); ++facei){
        const label own = owner[facei];
        test_unique_g[own]+=1;
    }

    //find maximum
    int max_repetition = 0;
    for (label i=0; i < g_len; ++i){
       max_repetition = max_repetition < test_unique_g[i] ? test_unique_g[i] :  max_repetition;
    }
    fprintf(stderr,"max_repetition in g[own] = %d owner.size() = %d\n",max_repetition,owner.size());

    for (label i=0; i < g_len; ++i) test_unique_g[i] = 0;
    for (label facei=0; facei < owner.size(); ++facei){
        const label nei = neighbour[facei];
        test_unique_g[nei]+=1;
    }
    max_repetition = 0;
    for (label i=0; i < g_len; ++i){
       max_repetition = max_repetition < test_unique_g[i] ? test_unique_g[i] :  max_repetition;
    }
    fprintf(stderr,"max_repetition in g[nei] = %d owner.size() = %d\n",max_repetition,owner.size());

    


    delete[] test_unique_g;

#endif

    #ifdef USE_ROCTX
    roctxRangePush("cellMDLimitedGrad::calcGrad:loop3");
    #endif

//    static unsigned char *lock_index = NULL;

//    if (lock_index == NULL){
//     lock_index = new unsigned char[g.size()];
//     for (label i = 0; i < g.size(); ++i) lock_index[i] = 0;
//    }

#if 1
    static label *offsets = NULL;
    static label *face_list = NULL; 

    if (face_list == NULL){
       fprintf(stderr, "setting up\n");

       offsets = new label[g.size()+1];
       label *count = new label[g.size()];

       for (label i = 0; i < g.size(); ++i ) count[i] = 0;

       

       for (label facei=0; facei < owner.size(); ++facei)
       {
        const label own = owner[facei];
        const label nei = neighbour[facei];
        count[own]++;
        count[nei]++;
       }
       
       offsets[0] = 0;
       for (label i = 0; i < g.size(); ++i ){
         offsets[i+1] = offsets[i]+count[i];
       } 
       face_list = new label[offsets[g.size()]];

       //list faces for each cell
       for (label i = 0; i < g.size(); ++i ) count[i] = 0;

       label *ptr_to_face_list;

       for (label facei=0; facei < owner.size(); ++facei){
     
        const label own = owner[facei];
        const label nei = neighbour[facei];

        ptr_to_face_list = &face_list[ offsets[own] + count[own] ];
        ptr_to_face_list[0] = facei;
        count[own]++;

        ptr_to_face_list = &face_list[ offsets[nei] + count[nei] ];
        ptr_to_face_list[0] = facei;
        count[nei]++;        
       }
       delete[] count;
    }

    label nCells = g.size();

    #pragma omp target teams distribute parallel for  if(nCells > 10000)
    for (label cell = 0; cell < nCells; ++cell){

        const label *ptr_to_face_list = &face_list[offsets[cell]];
        const label nFaces = offsets[cell+1] - offsets[cell];
        
        scalar mxV = maxVsf[cell];
        scalar mnV = minVsf[cell]; 
        auto C_cell = C[cell];
        auto g_cell = g[cell];
        #pragma unroll 2
        for ( label f = 0; f < nFaces; ++f){
            label facei = ptr_to_face_list[f];
            limitFace
            (
            g_cell,
            mxV,
            mnV,
            Cf[facei] - C_cell 
            );
        }
     }

#else



    const label loop_len2 = owner.size();
   // #pragma omp target teams distribute parallel for  if(loop_len2 > 20000)
    for (label facei=0; facei < loop_len2; ++facei)
    //forAll(owner, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        // owner side
        limitFace
        (
            g[own],
            maxVsf[own],
            minVsf[own],
            Cf[facei] - C[own] /*, lock_index[own]*/
        );

        // neighbour side
        limitFace
        (
            g[nei],
            maxVsf[nei],
            minVsf[nei],
            Cf[facei] - C[nei] /*, lock_index[nei]*/
        );
    }
#endif

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    #ifdef USE_ROCTX
    roctxRangePush("cellMDLimitedGrad::calcGrad:loop4");
    #endif

    forAll(bsf, patchi)
    {
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = Cf.boundaryField()[patchi];

	//AMD LG - is it safe ? better to loop over cells and faces within a cell
	const label loop_len = pOwner.size();
        #pragma omp target teams distribute parallel for  if(loop_len > 10000)
        for (label pFacei=0; pFacei < loop_len; ++pFacei)
        //forAll(pOwner, pFacei)
        {
            const label own = pOwner[pFacei];

            limitFace
            (
                g[own],
                maxVsf[own],
                minVsf[own],
                pCf[pFacei] - C[own]
            );
        }
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif



    g.correctBoundaryConditions();
    gaussGrad<scalar>::correctBoundaryConditions(vsf, g);

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tGrad;
}


template<>
Foam::tmp<Foam::volTensorField>
Foam::fv::cellMDLimitedGrad<Foam::vector>::calcGrad
(
    const volVectorField& vsf,
    const word& name
) const
{

    

    const fvMesh& mesh = vsf.mesh();

    tmp<volTensorField> tGrad = basicGradScheme_().calcGrad(vsf, name);

    if (k_ < SMALL)
    {
        return tGrad;
    }

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellMDLimitedGrad<vector>::calcGrad");
    #endif




    volTensorField& g = tGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const volVectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    vectorField maxVsf(vsf.primitiveField());
    vectorField minVsf(vsf.primitiveField());

    #if 1

      const label loop_len = owner.size();
      #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 5000) 
      for (label facei = 0; facei < loop_len; facei+=2){
	  
	const label nf = (loop_len-facei) > 1 ? 2 : 1;
 
	for (label i = 0; i < nf; ++i){
          const label own = owner[facei];
          const label nei = neighbour[facei];
          const Foam::Vector<scalar>& vsfOwn = vsf[own];
          const Foam::Vector<scalar>& vsfNei = vsf[nei];

          //maxVsf[own] = Foam::max(maxVsf[own], vsfNei);
          for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
            scalar& var = setComponent(maxVsf[own],cmpt);
            #pragma omp atomic compare            
            if (var < (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
          }

          //minVsf[own] = Foam::min(minVsf[own], vsfNei);
          for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
            scalar& var = setComponent(minVsf[own],cmpt);
            #pragma omp atomic compare
            if (var > (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
          }

          //maxVsf[nei] = Foam::max(maxVsf[nei], vsfOwn);
          for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
            scalar& var = setComponent(maxVsf[nei],cmpt);
            #pragma omp atomic compare
            if (var < (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
          }

          //minVsf[nei] = Foam::min(minVsf[nei], vsfOwn);
          for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
            scalar& var = setComponent(minVsf[nei],cmpt);
            #pragma omp atomic compare
            if (var > (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
          }
	}
      }
                       
    #else

    forAll(owner, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        const vector& vsfOwn = vsf[own];
        const vector& vsfNei = vsf[nei];

        maxVsf[own] = max(maxVsf[own], vsfNei);
        minVsf[own] = min(minVsf[own], vsfNei);

        maxVsf[nei] = max(maxVsf[nei], vsfOwn);
        minVsf[nei] = min(minVsf[nei], vsfOwn);
    }
    #endif


    const volVectorField::Boundary& bsf = vsf.boundaryField();

    forAll(bsf, patchi)
    {
        const fvPatchVectorField& psf = bsf[patchi];
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();

        if (psf.coupled())
        {
            const vectorField psfNei(psf.patchNeighbourField());

            #if 1
            const label loop_len = pOwner.size();
            #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 5000) 
            for (label pFacei = 0; pFacei < loop_len; ++pFacei){

	        const label own = pOwner[pFacei];
                const vector& vsfNei = psfNei[pFacei];
                 //maxVsf[own] = max(maxVsf[own], vsfNei);
                 for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
                    scalar& var = setComponent(maxVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (var < (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }
                 //minVsf[own] = min(minVsf[own], vsfNei);
                 for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
                    scalar& var = setComponent(minVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (var > (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }
	    }
	    #else
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                const vector& vsfNei = psfNei[pFacei];

                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
            }
            #endif
	}
        else
        {
	    #if 1
            const label loop_len = pOwner.size();
            #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 5000) 
            for (label pFacei = 0; pFacei < loop_len; ++pFacei){

                const label own = pOwner[pFacei];
                const vector& vsfNei = psf[pFacei];
                //maxVsf[own] = max(maxVsf[own], vsfNei);
                for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
                    scalar& var = setComponent(maxVsf[own],cmpt);
                    #pragma omp atomic compare
                     if (var < (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }

                //minVsf[own] = min(minVsf[own], vsfNei);
                for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
                    scalar& var = setComponent(minVsf[own],cmpt);
                    #pragma omp atomic compare
                     if (var > (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }
	    }

            #else		
            forAll(pOwner, pFacei)
            {
                const label own = pOwner[pFacei];
                const vector& vsfNei = psf[pFacei];

                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
            }
            #endif
        }
    }

    maxVsf -= vsf;
    minVsf -= vsf;

    if (k_ < 1.0)
    {
        const vectorField maxMinVsf((1.0/k_ - 1.0)*(maxVsf - minVsf));
        maxVsf += maxMinVsf;
        minVsf -= maxMinVsf;

        //maxVsf *= 1.0/k_;
        //minVsf *= 1.0/k_;
    }

#if 1
    static label *offsets = NULL;
    static label *face_list = NULL;

    if (face_list == NULL){
       fprintf(stderr, "setting up\n");

       offsets = new label[g.size()+1];
       label *count = new label[g.size()];

       for (label i = 0; i < g.size(); ++i ) count[i] = 0;



       for (label facei=0; facei < owner.size(); ++facei)
       {
        const label own = owner[facei];
        const label nei = neighbour[facei];
        count[own]++;
        count[nei]++;
       }

       offsets[0] = 0;
       for (label i = 0; i < g.size(); ++i ){
         offsets[i+1] = offsets[i]+count[i];
       }
       face_list = new label[offsets[g.size()]];

       //list faces for each cell
       for (label i = 0; i < g.size(); ++i ) count[i] = 0;

       label *ptr_to_face_list;

       for (label facei=0; facei < owner.size(); ++facei){

        const label own = owner[facei];
        const label nei = neighbour[facei];

        ptr_to_face_list = &face_list[ offsets[own] + count[own] ];
        ptr_to_face_list[0] = facei;
        count[own]++;

        ptr_to_face_list = &face_list[ offsets[nei] + count[nei] ];
        ptr_to_face_list[0] = facei;
        count[nei]++;
       }
       delete[] count;
    }

    label nCells = g.size();

    #pragma omp target teams distribute parallel for  if(nCells > 10000)
    for (label cell = 0; cell < nCells; ++cell){

        label *ptr_to_face_list = &face_list[offsets[cell]];
        const label nFaces = offsets[cell+1] - offsets[cell];

        auto mxV = maxVsf[cell];
        auto mnV = minVsf[cell];
        auto C_cell = C[cell];
        auto g_cell = g[cell];
        #pragma unroll 2
        for ( label f = 0; f < nFaces; ++f){
            label facei = ptr_to_face_list[f];
            limitFace
            (
            g_cell,
            mxV,
            mnV,
            Cf[facei] - C_cell
            );
        }
     }

#else


    forAll(owner, facei)
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        // owner side
        limitFace
        (
            g[own],
            maxVsf[own],
            minVsf[own],
            Cf[facei] - C[own]
        );

        // neighbour side
        limitFace
        (
            g[nei],
            maxVsf[nei],
            minVsf[nei],
            Cf[facei] - C[nei]
        );
    }
#endif

    forAll(bsf, patchi)
    {
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = Cf.boundaryField()[patchi];

        forAll(pOwner, pFacei)
        {
            const label own = pOwner[pFacei];

            limitFace
            (
                g[own],
                maxVsf[own],
                minVsf[own],
                pCf[pFacei] - C[own]
            );
        }
    }

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellMDLimitedGrad<vector>::calcGrad-boundary");
    #endif

    g.correctBoundaryConditions();
    gaussGrad<vector>::correctBoundaryConditions(vsf, g);

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tGrad;
}


// ************************************************************************* //
