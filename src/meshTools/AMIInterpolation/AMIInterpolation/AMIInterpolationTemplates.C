/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2015-2018 OpenCFD Ltd.
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

#include "profiling.H"
#include "mapDistribute.H"


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




// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class Type, class CombineOp>
void Foam::AMIInterpolation::interpolateToTarget
(
    const UList<Type>& fld,
    const CombineOp& cop,
    List<Type>& result,
    const UList<Type>& defaultValues
) const
{
    addProfiling(ami, "AMIInterpolation::interpolateToTarget");


    #ifdef USE_ROCTX
    roctxRangePush("AMIInterpolation::interpolateToTarget");
    #endif

    if (fld.size() != srcAddress_.size())
    {
        FatalErrorInFunction
            << "Supplied field size is not equal to source patch size" << nl
            << "    source patch   = " << srcAddress_.size() << nl
            << "    target patch   = " << tgtAddress_.size() << nl
            << "    supplied field = " << fld.size()
            << abort(FatalError);
    }

    if (lowWeightCorrection_ > 0)
    {
        if (defaultValues.size() != tgtAddress_.size())
        {
            FatalErrorInFunction
                << "Employing default values when sum of weights falls below "
                << lowWeightCorrection_
                << " but supplied default field size is not equal to target "
                << "patch size" << nl
                << "    default values = " << defaultValues.size() << nl
                << "    target patch   = " << tgtAddress_.size() << nl
                << abort(FatalError);
        }
    }

    result.setSize(tgtAddress_.size());

    if (distributed())
    {
        const mapDistribute& map = srcMapPtr_();

	#ifdef USE_ROCTX
        roctxRangePush("AMIInterpolation::interpolateToTarget_2");
        #endif

        List<Type> work(fld);

        #ifdef USE_ROCTX
        roctxRangePop();
        #endif

	#ifdef USE_ROCTX
        roctxRangePush("AMIInterpolation::interpolateToTarget_3");
        #endif

	map.distribute(work);


        #ifdef USE_ROCTX
        roctxRangePop();
        #endif

#if 1
if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ){

        const label loop_len = result.size();

        #pragma omp target teams distribute parellel for if(loop_len > 5000) 
        for (label facei = 0; facei < loop_len; ++facei)
        {
            if (tgtWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = tgtAddress_[facei];
                const scalarList& weights = tgtWeights_[facei];

                Type result_tmp  = vector::zero;
                Type& tmp = (Type&) result_tmp;

                //forAll(faces, i)
                const label loop_len2 = faces.size();
		#pragma unroll 4
		for (label i = 0; i < loop_len2; ++i)  //parallelize within a team 
                {
                    cop(tmp /* result[facei] */, facei, work[faces[i]], weights[i]);
		}
		result[facei] = result_tmp;
	    }
	}

}
else if constexpr ( std::is_same_v<Type,scalar> ){

        const label loop_len = result.size();
        #pragma omp target teams distribute parellel for if(loop_len > 5000)
        for (label facei = 0; facei < loop_len; ++facei)
        {
            if (tgtWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = tgtAddress_[facei];
                const scalarList& weights = tgtWeights_[facei];

                Type result_tmp = 0.0;
                Type& tmp = (Type&) result_tmp;

                //forAll(faces, i)
                const label loop_len2 = faces.size();
                #pragma unroll 4
                for (label i = 0; i < loop_len2; ++i)  //parallelize within a team
                {
                    cop(tmp /* result[facei] */, facei, work[faces[i]], weights[i]);
                }
                result[facei] = tmp;
            }
        }

}
else{
#endif
	const label loop_len = result.size();
	for (label facei = 0; facei < loop_len; ++facei)
        {
            if (tgtWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = tgtAddress_[facei];
                const scalarList& weights = tgtWeights_[facei];

                forAll(faces, i)
                {
                    cop(result[facei], facei, work[faces[i]], weights[i]);
                }
            }
        }
}

    }
    else
    {
	    
        //forAll(result, facei)
        const label loop_len = result.size();
        for (label facei = 0; facei < loop_len; ++facei)		
        {
            if (tgtWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = tgtAddress_[facei];
                const scalarList& weights = tgtWeights_[facei];

                forAll(faces, i)
                {
                    cop(result[facei], facei, fld[faces[i]], weights[i]);
                }
            }
        }
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif
}


template<class Type, class CombineOp>
void Foam::AMIInterpolation::interpolateToSource
(
    const UList<Type>& fld,
    const CombineOp& cop,
    List<Type>& result,
    const UList<Type>& defaultValues
) const
{
    addProfiling(ami, "AMIInterpolation::interpolateToSource");

    #ifdef USE_ROCTX
    roctxRangePush("AMIInterpolation::interpolateToSource");
    #endif


    if (fld.size() != tgtAddress_.size())
    {
        FatalErrorInFunction
            << "Supplied field size is not equal to target patch size" << nl
            << "    source patch   = " << srcAddress_.size() << nl
            << "    target patch   = " << tgtAddress_.size() << nl
            << "    supplied field = " << fld.size()
            << abort(FatalError);
    }

    if (lowWeightCorrection_ > 0)
    {
        if (defaultValues.size() != srcAddress_.size())
        {
            FatalErrorInFunction
                << "Employing default values when sum of weights falls below "
                << lowWeightCorrection_
                << " but supplied default field size is not equal to source "
                << "patch size" << nl
                << "    default values = " << defaultValues.size() << nl
                << "    source patch   = " << srcAddress_.size() << nl
                << abort(FatalError);
        }
    }

    result.setSize(srcAddress_.size());

    if (distributed())
    {
	//fprintf(stderr,"AMIInterpolation::interpolateToSource in distributed\n");    
        const mapDistribute& map = tgtMapPtr_();

        List<Type> work(fld);
        map.distribute(work);


        if constexpr ( std::is_same_v<Type,Foam::Vector<scalar>> ){
          const label loop_len = result.size();
          #pragma omp target teams distribute parallel for thread_limit(64)  if (loop_len > 5000) 
          for (label facei = 0; facei < loop_len; ++facei)
          {
            if (srcWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = srcAddress_[facei];
                const scalarList& weights = srcWeights_[facei];

                Type result_tmp  = vector::zero;
		Type& tmp = (Type&) result_tmp;

                //forAll(faces, i)
		const label loop_len2 = faces.size();
	        #pragma unroll 4
	        for (label i = 0; i < loop_len2; ++i)	
                {
                    cop( tmp /*result[facei]*/, facei, work[faces[i]], weights[i]);
                }
		result[facei] = result_tmp;
            }
          }
        }
	else if constexpr ( std::is_same_v<Type,scalar> ){
          const label loop_len = result.size();
          #pragma omp target teams distribute parallel for thread_limit(64)  if (loop_len > 5000)
          for (label facei = 0; facei < loop_len; ++facei)
          {
            if (srcWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = srcAddress_[facei];
                const scalarList& weights = srcWeights_[facei];

                Type result_tmp  = 0.0;
                Type& tmp = (Type&) result_tmp;

                //forAll(faces, i)
                const label loop_len2 = faces.size();
                #pragma unroll 4
                for (label i = 0; i < loop_len2; ++i)
                {
                    cop( tmp /*result[facei]*/, facei, work[faces[i]], weights[i]);
                }
                result[facei] = result_tmp;
            }
          }
        }

	else
	{
//	fprintf(stderr,"in file=%s in line=%d, 	typeid(Type).name()=%s\n",__FILE__, __LINE__, typeid(Type).name());
          //forAll(result, facei)
          const label loop_len = result.size();
          #pragma omp parallel for if(loop_len > 500)
          for (label facei = 0; facei < loop_len; ++facei)
          {
            if (srcWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = srcAddress_[facei];
                const scalarList& weights = srcWeights_[facei];

                forAll(faces, i)
                {
                    cop(result[facei], facei, work[faces[i]], weights[i]);
                }
            }
          }
	}
    }
    else
    {
        //fprintf(stderr,"AMIInterpolation::interpolateToSource not in distributed\n");

        forAll(result, facei)
        {
            if (srcWeightsSum_[facei] < lowWeightCorrection_)
            {
                result[facei] = defaultValues[facei];
            }
            else
            {
                const labelList& faces = srcAddress_[facei];
                const scalarList& weights = srcWeights_[facei];

                forAll(faces, i)
                {
                    cop(result[facei], facei, fld[faces[i]], weights[i]);
                }
            }
        }
    }
    #ifdef USE_ROCTX
    roctxRangePop();
    #endif
}


template<class Type, class CombineOp>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToSource
(
    const Field<Type>& fld,
    const CombineOp& cop,
    const UList<Type>& defaultValues
) const
{
    auto tresult = tmp<Field<Type>>::New(srcAddress_.size(), Zero);

    interpolateToSource
    (
        fld,
        multiplyWeightedOp<Type, CombineOp>(cop),
        tresult.ref(),
        defaultValues
    );

    return tresult;
}


template<class Type, class CombineOp>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToSource
(
    const tmp<Field<Type>>& tFld,
    const CombineOp& cop,
    const UList<Type>& defaultValues
) const
{
    return interpolateToSource(tFld(), cop, defaultValues);
}


template<class Type, class CombineOp>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToTarget
(
    const Field<Type>& fld,
    const CombineOp& cop,
    const UList<Type>& defaultValues
) const
{
    auto tresult = tmp<Field<Type>>::New(tgtAddress_.size(), Zero);

    interpolateToTarget
    (
        fld,
        multiplyWeightedOp<Type, CombineOp>(cop),
        tresult.ref(),
        defaultValues
    );

    return tresult;
}


template<class Type, class CombineOp>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToTarget
(
    const tmp<Field<Type>>& tFld,
    const CombineOp& cop,
    const UList<Type>& defaultValues
) const
{
    return interpolateToTarget(tFld(), cop, defaultValues);
}


template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToSource
(
    const Field<Type>& fld,
    const UList<Type>& defaultValues
) const
{
    return interpolateToSource(fld, plusEqOp<Type>(), defaultValues);
}


template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToSource
(
    const tmp<Field<Type>>& tFld,
    const UList<Type>& defaultValues
) const
{
    return interpolateToSource(tFld(), plusEqOp<Type>(), defaultValues);
}


template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToTarget
(
    const Field<Type>& fld,
    const UList<Type>& defaultValues
) const
{
    return interpolateToTarget(fld, plusEqOp<Type>(), defaultValues);
}


template<class Type>
Foam::tmp<Foam::Field<Type>> Foam::AMIInterpolation::interpolateToTarget
(
    const tmp<Field<Type>>& tFld,
    const UList<Type>& defaultValues
) const
{
    return interpolateToTarget(tFld(), plusEqOp<Type>(), defaultValues);
}


// ************************************************************************* //
