/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018 OpenFOAM Foundation
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

#include "cellLimitedGrad.H"
#include "gaussGrad.H"

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

#include <type_traits>


#define USM_Cell_Limit_Grad

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type, class Limiter>
void Foam::fv::cellLimitedGrad<Type, Limiter>::limitGradient
(
    const Field<scalar>& limiter,
    Field<vector>& gIf
) const
{
    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_A");
    #endif

    gIf *= limiter;

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

}


template<class Type, class Limiter>
void Foam::fv::cellLimitedGrad<Type, Limiter>::limitGradient
(
    const Field<vector>& limiter,
    Field<tensor>& gIf
) const
{
    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_B");
    #endif


    //forAll(gIf, celli)
    #pragma omp target teams distribute parallel for if(gIf.size() > 20000) //LG4 tested, OK
    for (label celli=0; celli < gIf.size(); ++celli)
    {
        gIf[celli] = tensor
        (
            cmptMultiply(limiter[celli], gIf[celli].x()),
            cmptMultiply(limiter[celli], gIf[celli].y()),
            cmptMultiply(limiter[celli], gIf[celli].z())
        );
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif
}

template< class T >
void test_type(){
   if (std::is_same_v<T, double>) printf("test_type: T==double\n");
   if (std::is_same_v<T, float>) printf("test_type: T==float\n");
   if (std::is_same_v<T,Foam::Vector<double>>) printf("test_type: T==Foam::Vector<double>\n");
   if (std::is_same_v<T,Foam::Vector<float>>) printf("test_type: T==Foam::Vector<float>\n");
}


template<class Type, class Limiter>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::cellLimitedGrad<Type, Limiter>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{

    //test_type<Type>();

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_C");
    #endif

    const fvMesh& mesh = vsf.mesh();

    tmp
    <
        GeometricField
        <typename outerProduct<vector, Type>::type, fvPatchField, volMesh>
    > tGrad = basicGradScheme_().calcGrad(vsf, name);

    if (k_ < SMALL)
    {
        #ifdef USE_ROCTX
        roctxRangePop();
        #endif
        return tGrad;
    }

    GeometricField
    <
        typename outerProduct<vector, Type>::type,
        fvPatchField,
        volMesh
    >& g = tGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    const volVectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    Field<Type> maxVsf(vsf.primitiveField());
    Field<Type> minVsf(vsf.primitiveField());

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_min_max");
    #endif



#if 1
    static label *offsets = NULL;
    static label *neighbour_list = NULL;

    if (neighbour_list == NULL){
       fprintf(stderr, "CELLLIMITEDGRAD: setting up\n");

       offsets = new label[maxVsf.size()+1];
       label *count = new label[maxVsf.size()];

       for (label i = 0; i < maxVsf.size(); ++i ) count[i] = 0;

       //count neighbours for each owner
       for (label facei=0; facei < owner.size(); ++facei)
       {
        const label own = owner[facei];
        count[own]++;
       }

       offsets[0] = 0;
       for (label i = 0; i < maxVsf.size(); ++i ){
         offsets[i+1] = offsets[i]+count[i];
       }
       neighbour_list = new label[offsets[maxVsf.size()]];

       //list faces for each cell
       for (label i = 0; i < maxVsf.size(); ++i ) count[i] = 0;

       label *ptr_to_neighbour_list;

       for (label facei=0; facei < owner.size(); ++facei){
        const label own = owner[facei];
        const label nei = neighbour[facei];
        ptr_to_neighbour_list = &neighbour_list[ offsets[own] + count[own] ];
        ptr_to_neighbour_list[0] = nei;
        count[own]++;
       }
       delete[] count;
    }


     if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> || std::is_same_v<Type,scalar> ) {
        label loop_len = maxVsf.size();
        #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 10000) 
        for (label celli = 0; celli < loop_len; celli+=1){

  	  const label *ptr_to_neighbour_list = &neighbour_list[offsets[celli]];
          const label nFaces = offsets[celli+1] - offsets[celli];

          //Foam::Vector<scalar> maxVsf_celli = maxVsf[celli];
	  //Foam::Vector<scalar> minVsf_celli = minVsf[celli];
	  Type maxVsf_celli = maxVsf[celli];
	  Type minVsf_celli = minVsf[celli];
          #pragma unroll 2
          for ( label f = 0; f < nFaces; ++f){
            label nei = ptr_to_neighbour_list[f];
            maxVsf_celli = Foam::max(maxVsf_celli, vsf[nei]);
            minVsf_celli = Foam::min(minVsf_celli, vsf[nei]);
	  }
	  maxVsf[celli] = maxVsf_celli;
	  minVsf[celli] = minVsf_celli;
        }


        loop_len = owner.size();
        #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 10000) 
        for (label facei = 0; facei < loop_len; facei += 2){

          const label nf = (loop_len-facei) > 1 ? 2 : 1;
          #pragma unroll 2
          for ( label i = 0; i < nf; ++i){

            const label own = owner[facei+i];
            const label nei = neighbour[facei+i];
            

	    const Type& vsfOwn = vsf[own];
            const Type& vsfNei = vsf[nei];


            //if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ){

              //maxVsf[nei] = Foam::max(maxVsf[nei], vsfOwn);
              for (direction cmpt = 0; cmpt < pTraits<Type>::nComponents; ++cmpt){
                scalar& var = setComponent(maxVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var < (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
              }

              //minVsf[nei] = Foam::min(minVsf[nei], vsfOwn);
              for (direction cmpt = 0; cmpt < pTraits<Type>::nComponents; ++cmpt){
                scalar& var = setComponent(minVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var > (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
              }
	    //}
#if 0
	    if    constexpr ( std::is_same_v<Type,scalar> ) {
              for (direction cmpt=0; cmpt<pTraits<scalar>::nComponents; ++cmpt){
                scalar& var = setComponent(maxVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var < (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
              }
              for (direction cmpt=0; cmpt<pTraits<scalar>::nComponents; ++cmpt){
                scalar& var = setComponent(minVsf[nei],cmpt);
                #pragma omp atomic compare
                if (var > (scalar) component(vsfOwn,cmpt)) var = (scalar) component(vsfOwn,cmpt);
              }
	    }
#endif
          }
        }
     }
     else{

       fprintf(stderr,"not a type for offloading , Type is  %s length = %d, line=%d file=%s\n", owner.size(), typeid(Type).name(),  __LINE__, __FILE__);


      for (label facei = 0; facei < owner.size(); ++facei){
        const label own = owner[facei];
        const label nei = neighbour[facei];
        const Type& vsfOwn = vsf[own];
        const Type& vsfNei = vsf[nei];

        maxVsf[own] = Foam::max(maxVsf[own], vsfNei);
        minVsf[own] = Foam::min(minVsf[own], vsfNei);
        maxVsf[nei] = Foam::max(maxVsf[nei], vsfOwn);
        minVsf[nei] = Foam::min(minVsf[nei], vsfOwn);
      }
    }



    #else



    if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ) {   
      const label loop_len = owner.size(); 
      #pragma omp target teams distribute parallel for thread_limit(256) if(loop_len > 10000) 
      for (label facei = 0; facei < loop_len; facei += 2){

        const label nf = (loop_len-facei) > 1 ? 2 : 1;
        #pragma unroll 2
        for ( label i = 0; i < nf; ++i){

          const label own = owner[facei+i];
          const label nei = neighbour[facei+i];
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
    }
    else{
      	    
      for (label facei = 0; facei < owner.size(); ++facei){
        const label own = owner[facei];
        const label nei = neighbour[facei];
        const Type& vsfOwn = vsf[own];
        const Type& vsfNei = vsf[nei];
        
	maxVsf[own] = Foam::max(maxVsf[own], vsfNei);
        minVsf[own] = Foam::min(minVsf[own], vsfNei);
        maxVsf[nei] = Foam::max(maxVsf[nei], vsfOwn);
        minVsf[nei] = Foam::min(minVsf[nei], vsfOwn);
      }
    }
    #endif

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_boundary_min_max");
    #endif

    const auto& bsf = vsf.boundaryField();


    for (label patchi=0; patchi < bsf.size(); ++patchi)
    {
        const fvPatchField<Type>& psf = bsf[patchi];
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();

       // printf("patchi=%d :  pOwner.size()=%d\n",patchi,pOwner.size());

        if (psf.coupled())
        {
            const Field<Type> psfNei(psf.patchNeighbourField());

	    const label loop_len = pOwner.size();
            //if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ) {
            if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> || std::is_same_v<Type,scalar> ) {
              #pragma omp target teams distribute parallel for if(loop_len > 10000) 
              for (label pFacei = 0; pFacei < loop_len; ++pFacei){
		 const label own = pOwner[pFacei];
                 const Type& vsfNei = psfNei[pFacei];

                 //maxVsf[own] = max(maxVsf[own], vsfNei);
		 //for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
		for (direction cmpt=0; cmpt<pTraits<Type>::nComponents; ++cmpt){
                    scalar& var = setComponent(maxVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (var < (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }
                 //minVsf[own] = min(minVsf[own], vsfNei);
		 //for (direction cmpt=0; cmpt<pTraits<Foam::Vector<scalar>>::nComponents; ++cmpt){
		 for (direction cmpt=0; cmpt<pTraits<Type>::nComponents; ++cmpt){
                    scalar& var = setComponent(minVsf[own],cmpt);
                    #pragma omp atomic compare
                    if (var > (scalar) component(vsfNei,cmpt)) var = (scalar) component(vsfNei,cmpt);
                 }
	       }
	    }
	    else {

              forAll(pOwner, pFacei)
              {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psfNei[pFacei];

                //atomic MAX/MIN should solve the race condition issue 
                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
              }
	    }
        }
        else
        {
	    const label loop_len = pOwner.size(); 	
            //if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ) {
            if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> || std::is_same_v<Type,scalar> ) {		    
              #pragma omp target teams distribute parallel for if(loop_len > 10000) 
              for (label pFacei = 0; pFacei < loop_len; ++pFacei){
                const label own = pOwner[pFacei];
                const Type& vsfNei = psf[pFacei];
                
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
	    }
	    else
	    {

              forAll(pOwner, pFacei)             
              {
                const label own = pOwner[pFacei];
                const Type& vsfNei = psf[pFacei];
                maxVsf[own] = max(maxVsf[own], vsfNei);
                minVsf[own] = min(minVsf[own], vsfNei);
              }
	    }
        }
     }

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_C:update");
    #endif

    maxVsf -= vsf;
    minVsf -= vsf;

    if (k_ < 1.0)
    {
        const Field<Type> maxMinVsf((1.0/k_ - 1.0)*(maxVsf - minVsf));
        maxVsf += maxMinVsf;
        minVsf -= maxMinVsf;
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


    // Create limiter initialized to 1
    // Note: the limiter is not permitted to be > 1
    Field<Type> limiter(vsf.primitiveField().size(), pTraits<Type>::one);

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_C:limitFace");
    #endif
   
    #if 0
    forAll(owner, facei)
    #else
    #pragma omp target teams distribute parallel for  if(owner.size() > 10000)
    for (label facei=0; facei < owner.size(); ++facei)    
    #endif
    {
        const label own = owner[facei];
        const label nei = neighbour[facei];

        // owner side
        limitFace
        (
            limiter[own],
            maxVsf[own],
            minVsf[own],
            (Cf[facei] - C[own]) & g[own]
        );

        // neighbour side
        limitFace
        (
            limiter[nei],
            maxVsf[nei],
            minVsf[nei],
            (Cf[facei] - C[nei]) & g[nei]
        );
    }

    forAll(bsf, patchi)
    {
        const labelUList& pOwner = mesh.boundary()[patchi].faceCells();
        const vectorField& pCf = Cf.boundaryField()[patchi];

        #if 0
        forAll(pOwner, pFacei)
        #else
        #pragma omp target teams distribute parallel for if(owner.size() > 10000)
        for (label pFacei = 0; pFacei < pOwner.size(); ++pFacei)
        #endif
        {
            const label own = pOwner[pFacei]; 

            limitFace
            (
                limiter[own],
                maxVsf[own],
                minVsf[own],
                ((pCf[pFacei] - C[own]) & g[own])
            );
        }
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    if (fv::debug)
    {
        Info<< "gradient limiter for: " << vsf.name()
            << " max = " << gMax(limiter)
            << " min = " << gMin(limiter)
            << " average: " << gAverage(limiter) << endl;
    }

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_C:limitGradient");
    #endif

    limitGradient(limiter, g);
    
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    #ifdef USE_ROCTX
    roctxRangePush("fv::cellLimitedGrad_C:correctBoundaryConditions");
    #endif
    g.correctBoundaryConditions();
    gaussGrad<Type>::correctBoundaryConditions(vsf, g);
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif


    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tGrad;
}


// ************************************************************************* //
