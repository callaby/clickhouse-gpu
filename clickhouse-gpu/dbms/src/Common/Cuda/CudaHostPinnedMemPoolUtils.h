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

class CudaHostPinnedMemPoolUtils 
{
public:
    static const std::size_t CalculatePadding(const std::size_t baseAddress, const std::size_t alignment) 
    {
        const std::size_t multiplier = (baseAddress / alignment) + 1;
        const std::size_t alignedAddress = multiplier * alignment;
        const std::size_t padding = alignedAddress - baseAddress;
        return padding;
    }

    static const std::size_t CalculatePaddingWithHeader(const std::size_t baseAddress, const std::size_t alignment, const std::size_t headerSize) 
    {
        std::size_t padding = CalculatePadding(baseAddress, alignment);
        std::size_t neededSpace = headerSize;

        if (padding < neededSpace){
            // Header does not fit - Calculate next aligned address that header fits
            neededSpace -= padding;

            // How many alignments I need to fit the header        
            if(neededSpace % alignment > 0){
                padding += alignment * (1+(neededSpace / alignment));
            }else {
                padding += alignment * (neededSpace / alignment);
            }
        }

        return padding;
    }
};
