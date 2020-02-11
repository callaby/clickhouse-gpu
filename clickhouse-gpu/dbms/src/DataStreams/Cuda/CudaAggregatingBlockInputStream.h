// Copyright 2016-2020 NVIDIA
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <Interpreters/Context.h>
#include <Interpreters/Cuda/CudaAggregator.h>
#include <IO/ReadBufferFromFile.h>
#include <IO/CompressedReadBuffer.h>
#include <DataStreams/IProfilingBlockInputStream.h>


namespace DB
{


/** See description of AggregatingBlockInputStream
  * 
  * 
  * 
  */
class CudaAggregatingBlockInputStream : public IProfilingBlockInputStream
{
public:
    /** keys are taken from the GROUP BY part of the query
      * Aggregate functions are searched everywhere in the expression.
      * Columns corresponding to keys and arguments of aggregate functions must already be computed.
      */
    CudaAggregatingBlockInputStream(const BlockInputStreamPtr & input, const Aggregator::Params & params_, 
      const Context & context_, bool final_)
        : params(params_), aggregator(context_, params), final(final_)
    {
        children.push_back(input);
    }

    String getName() const override { return "CudaAggregating"; }

    Block getHeader() const override;

protected:
    Block readImpl() override;

    CudaAggregator::Params params;
    CudaAggregator aggregator;
    bool final;

    bool executed = false;

    /** From here we will get the completed blocks after the aggregation. */
    BlocksList blocks;
    //std::unique_ptr<IBlockInputStream> impl;

    Logger * log = &Logger::get("CudaAggregatingBlockInputStream");
};

}
