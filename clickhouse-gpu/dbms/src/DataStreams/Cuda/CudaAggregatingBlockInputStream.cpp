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

#include <Common/ClickHouseRevision.h>

#include <DataStreams/BlocksListBlockInputStream.h>
#include <DataStreams/MergingAggregatedMemoryEfficientBlockInputStream.h>
#include <DataStreams/Cuda/CudaAggregatingBlockInputStream.h>
#include <DataStreams/NativeBlockInputStream.h>


namespace ProfileEvents
{
    extern const Event ExternalAggregationMerge;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

Block CudaAggregatingBlockInputStream::getHeader() const
{
    return aggregator.getHeader(final);
}


Block CudaAggregatingBlockInputStream::readImpl()
{
    if (!executed)
    {
        executed = true;
        CudaAggregatedDataVariantsPtr data_variants = std::make_shared<CudaAggregatedDataVariants>();

        //Aggregator::CancellationHook hook = [&]() { return this->isCancelled(); };
        //aggregator.setCancellationHook(hook);

        aggregator.execute(children.back(), *data_variants);

        blocks = aggregator.convertToBlocks(*data_variants, final, 1);
    }

    //if (isCancelledOrThrowIfKilled() || !impl)
    //    return {};

    if (blocks.empty())
        return Block();

    if (blocks.size() == 1)
    {
        Block res = blocks.back();
        blocks.clear();
        return res;
    }

    throw Exception("CudaAggregatingBlockInputStream::readImpl: blocks.size() is greater then 1", 
        ErrorCodes::LOGICAL_ERROR);
}


}
