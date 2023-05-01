#ifndef _KORANGAR_QUEUE_H_
#define _KORANGAR_QUEUE_H_

#include "korangar.h"
#include "util.h"

typedef struct
{
  kr_size size;
  kr_size capacity;
  u8      data[1];
} kr_queue;

#ifdef __cplusplus
extern "C" {
#endif

void kr_queue_expand(void** address, kr_size capacity);

#ifdef __cplusplus
}
#endif

#define kr_queue_release(address)                                                                                 \
  \
{                                                                                                                 \
	void* mem = (u8*)address - kr_offsetof(kr_queue, data);\
    kr_free(&mem);                                     \
    address = kr_null;                                                                                               \
}

#define kr_queue_reserve(address, capacity)                                                                          \
  \
{                                                                                                                 \
address = kr_null;                                                                                                   \
kr_queue_expand((void**)&address, sizeof(*address) * capacity);                                         \
    \
kr_queue_size(address)            = capacity;                                                                        \
  \
}

#define kr_queue_init(address, capacity)                                                                             \
  \
do {                                                                                                                 \
    \
address = kr_null;                                                                                                   \
    kr_queue_expand((void**)&address, sizeof(*address) * capacity);                                                  \
  \
} while(0)

#define kr_queue_size(address)                                                                                       \
  \
((kr_queue*)((u8*)address - kr_offsetof(kr_queue, data)))->size

#define kr_queue_capacity(address)                                                                                   \
  \
(((kr_queue*)((u8*)address - kr_offsetof(kr_queue, data)))->capacity / sizeof(*address))

#define kr_queue_last(address) ((kr_queue_size(address)) ? address[kr_queue_size(address) - 1] : kr_null)

#define kr_queue_remove(address)                                                                                     \
  if (kr_queue_size(address))                                                                                        \
  kr_queue_size(address)--


#define kr_queue_push(address, ...)                                                                                \
  do {                                                                                                                    \
    \
if(kr_queue_size(address) == kr_queue_capacity(address))                                                           \
    {                                                                                                                  \
      \
kr_queue_expand((void**)&address, sizeof(*address) * kr_queue_capacity(address) * 2);                              \
    \
}                                                                                                               \
address[kr_queue_size(address)++] = __VA_ARGS__ ;                                                                           \
\
} while(0)

#define kr_queue_grow(address, capacity)                                                                                \
  do {                                                                                                                    \
    \
if(kr_queue_capacity(address) - kr_queue_size(address) < (capacity))                                                           \
    {                                                                                                                  \
      \
kr_queue_expand((void**)&address, sizeof(*address) * (kr_queue_capacity(address) + capacity));                              \
    \
            kr_queue_size(treelets) += capacity;\
}                                                                                                               \
} while(0)

#endif /* _KORANGAR_QUEUE_H_ */
