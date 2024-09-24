/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2018-2021 OpenCFD Ltd.
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

#include "gaussGrad.H"
#include "extrapolatedCalculatedFvPatchField.H"
#include "AtomicAccumulator.H"
#include "macros.H"

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

template<class Type>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::gaussGrad<Type>::gradf
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf,
    const word& name
)
{
    #ifdef USE_ROCTX
    roctxRangePush("fv::gaussGrad_A");
    #endif



    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = ssf.mesh();

    tmp<GradFieldType> tgGrad
    (
        new GradFieldType
        (
            IOobject
            (
                name,
                ssf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<GradType>(ssf.dimensions()/dimLength, Zero),
            extrapolatedCalculatedFvPatchField<GradType>::typeName
        )
    );
    GradFieldType& gGrad = tgGrad.ref();

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const vectorField& Sf = mesh.Sf();

    Field<GradType>& igGrad = gGrad;
    const Field<Type>& issf = ssf;

    label loop_len = owner.size();
    //forAll(owner, facei)
    //
#if 1
    #pragma omp target teams distribute parallel for  if(loop_len>10000 )
    for (label facei = 0; facei <  loop_len; facei+=2) 
    {
	const label nf = (loop_len-facei) > 1 ? 2 : 1;
        #pragma unroll 2
        for ( uint32_t i = 0; i < nf; ++i){    
          const GradType Sfssf = Sf[facei+i]*issf[facei+i];
          atomicAccumulator(igGrad[owner[facei+i]]) += Sfssf;
          atomicAccumulator(igGrad[neighbour[facei+i]]) -= Sfssf;
	}
    }
#else


    static label *offsets = NULL;
    static label *face_list = NULL;
    static label *face_sign = NULL;



    if (face_list == NULL){
       fprintf(stderr, " GAUSS GRAD: setting up\n");

       offsets = new label[igGrad.size()+1];
       label *count = new label[igGrad.size()];

       for (label i = 0; i < igGrad.size(); ++i ) count[i] = 0;

       for (label facei=0; facei < owner.size(); ++facei)
       {
        const label own = owner[facei];
        const label nei = neighbour[facei];
        count[own]++;
        count[nei]++;
       }

       offsets[0] = 0;
       for (label i = 0; i < igGrad.size(); ++i ){
         offsets[i+1] = offsets[i]+count[i];
       }
       face_list = new label[offsets[igGrad.size()]];
       face_sign = new label[offsets[igGrad.size()]];

       //list faces for each cell
       for (label i = 0; i < igGrad.size(); ++i ) count[i] = 0;

       label *ptr_to_face_list, *ptr_to_face_sign;

       for (label facei=0; facei < owner.size(); ++facei){

        const label own = owner[facei];
        const label nei = neighbour[facei];

	/* face_list  has pairs [own] [nei]   this can be used to determine a sign
	 * for accumulating Sfssf   */
        ptr_to_face_list = &face_list[ offsets[own] + count[own] ];
	ptr_to_face_sign = &face_sign[ offsets[own] + count[own] ];
        ptr_to_face_list[0] = facei;
	ptr_to_face_sign[0] = 1.0;
        count[own]++;

        ptr_to_face_list = &face_list[ offsets[nei] + count[nei] ];
	ptr_to_face_sign = &face_sign[ offsets[nei] + count[nei] ];
        ptr_to_face_list[0] = facei;
	ptr_to_face_sign[0] = -1.0;
        count[nei]++;
       }
       delete[] count;
    
       label nCells = igGrad.size();


       const cellList& cells = mesh.cells();

       fprintf(stderr,"owner.size() = %d\n",owner.size());

       for (label celli = 0; celli < 2; ++celli){

         fprintf(stderr,"celli  = %d\n",celli);

         for (label f = 0; f < mesh.cells()[celli].size(); ++f){
              fprintf(stderr,"%d\t", mesh.cells()[celli][f]);
         }
         fprintf(stderr,"\n");

         label *ptr_to_face_list = &face_list[offsets[celli]];
         const label nFaces = offsets[celli+1] - offsets[celli];
         for ( label f = 0; f < nFaces; ++f){
           label facei = ptr_to_face_list[f];
           fprintf(stderr,"%d\t",facei);
         }
         fprintf(stderr,"\n");
       }
    }
    
    //double t1 = omp_get_wtime();

    #if 0
    const label nCells = igGrad.size();
    #pragma omp target teams distribute parallel for thread_limit(256) if(nCells>10000 )
    for (label celli = 0; celli < nCells; ++celli){

        const label *ptr_to_face_list = &face_list[offsets[celli]];
	const label *ptr_to_face_sign = &face_sign[offsets[celli]]; 
        const label nFaces = offsets[celli+1] - offsets[celli];

        #pragma unroll 2
        for ( label f = 0; f < nFaces; ++f){
	   const label facei = ptr_to_face_list[f];
	   const GradType Sfssf = Sf[facei]*issf[facei]*ptr_to_face_sign[f];
           igGrad[celli] += Sfssf;
	}
    }
    /*
    for (label celli = 0; celli < nCells; ++celli){

       for (label f = 0; f < mesh.cells()[celli].size(); ++f){
          const label facei = mesh.cells()[celli][f];
	  const GradType Sfssf = Sf[facei]*issf[facei];
          igGrad[celli] += Sfssf; //how to subtruct for neighbour ?
       }
    }
    */

    #else
    #pragma omp target teams distribute parallel for  if(loop_len>10000 )
    for (label facei = 0; facei <  loop_len; facei+=4)
    {
        const label nf = (loop_len-facei) > 3 ? 4 : loop_len-facei;
        #pragma unroll 4
        for ( label i = 0; i < nf; ++i){
          const GradType Sfssf = Sf[facei+i]*issf[facei+i];
          atomicAccumulator(igGrad[owner[facei+i]]) += Sfssf;
          atomicAccumulator(igGrad[neighbour[facei+i]]) -= Sfssf;
        }
    }
    #endif

    //double t2 = omp_get_wtime();
    //fprintf(stderr,"rank = %d, nCells = %d loop time  = %g\n",Pstream::myProcNo(), igGrad.size(), (t2-t1));

#endif


    label   mesh_boundary_size =  mesh.boundary().size();
    //forAll(mesh.boundary(), patchi)
    //#pragma omp target teams distribute 
    for (label patchi=0; patchi < mesh_boundary_size; ++patchi)
    {
        const labelUList& pFaceCells =
            mesh.boundary()[patchi].faceCells();

        const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

        const fvsPatchField<Type>& pssf = ssf.boundaryField()[patchi];

        //forAll(mesh.boundary()[patchi], facei)
        label mesh_boundary_patch_size = mesh.boundary()[patchi].size();
         #pragma omp target teams distribute parallel for if (mesh_boundary_patch_size>10000)
        for (label facei = 0; facei < mesh_boundary_patch_size; ++facei) 
        {
            atomicAccumulator(igGrad[pFaceCells[facei]]) += pSf[facei]*pssf[facei];
        }
    }

    igGrad /= mesh.V();

    gGrad.correctBoundaryConditions();


    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tgGrad;
}


template<class Type>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::gaussGrad<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{

    #ifdef USE_ROCTX
    roctxRangePush("fv::gaussGrad_B");
    #endif

    
    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    tmp<GradFieldType> tgGrad
    (
        gradf(tinterpScheme_().interpolate(vsf), name)
    );
    GradFieldType& gGrad = tgGrad.ref();

    correctBoundaryConditions(vsf, gGrad);

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

    return tgGrad;
}


template<class Type>
void Foam::fv::gaussGrad<Type>::correctBoundaryConditions
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    GeometricField
    <
        typename outerProduct<vector, Type>::type, fvPatchField, volMesh
    >& gGrad
)
{
    #ifdef USE_ROCTX
    roctxRangePush("fv::gaussGrad_C");
    #endif
    
    auto& gGradbf = gGrad.boundaryFieldRef();

    forAll(vsf.boundaryField(), patchi)
    {
        if (!vsf.boundaryField()[patchi].coupled())
        {
            const vectorField n
            (
                vsf.mesh().Sf().boundaryField()[patchi]
              / vsf.mesh().magSf().boundaryField()[patchi]
            );

            gGradbf[patchi] += n *
            (
                vsf.boundaryField()[patchi].snGrad()
              - (n & gGradbf[patchi])
            );
        }
     }

    #ifdef USE_ROCTX
    roctxRangePop();
    #endif

}


// ************************************************************************* //
