#pragma once
/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* This file implements methods to extract metadata from an integer Metadata field passed in as a template parameter. Feel free to add additional fields below.*/

#define RCCL_METADATA_EMPTY 0
#define RCCL_ONE_NODE_RING_SIMPLE (1 << 0)

constexpr bool isOneNodeRingSimple(int metadata) {
  return (metadata & RCCL_ONE_NODE_RING_SIMPLE) != 0;
}

static_assert(isOneNodeRingSimple(RCCL_ONE_NODE_RING_SIMPLE), "RCCL_ONE_NODE_RING_SIMPLE should be set to (1 << 0)");
static_assert(isOneNodeRingSimple(0) == 0, "RCCL_ONE_NODE_RING_SIMPLE should not be set when metadata is 0");