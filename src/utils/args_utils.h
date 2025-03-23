#ifndef _ARGS_UTILS_H_
#define _ARGS_UTILS_H_

#include "args_utils.h"
#include "types.h"

InputServerConfig parseServerArgs(int argc, char **argv, char **envp);

InputConfig parseInferArgs(int argc, char **argv, char **envp);

#endif
