/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/**
 * Authors: Alberto Dequino
*/ 

struct Embedding_args_fp16{
    fp16* BUFF;
    int dim;
    int embed_dim;
    int *ids;
    fp16 *embeds;
    fp16 *out;
};

void embedding_fw_tiled_fp16(void *embedding_args);