#include "queue.h"

void
kr_queue_expand(void** address, kr_size capacity)
{
  if (0 == capacity) return;

  kr_queue* queue;
  void* address_deref = *address;
  kr_queue* address_derefq = (kr_queue*)((u8*)address_deref - kr_offsetof(kr_queue, data));
  if (address_deref == kr_null) {
    queue       = (kr_queue*)kr_allocate(kr_offsetof(kr_queue, data) + capacity);
    kr_zero_memory(queue, kr_offsetof(kr_queue, data) + capacity);
    queue->size = 0;
  } else {
    queue = (kr_queue*)kr_reallocate(((u8*)address_deref - kr_offsetof(kr_queue, data)),
                                           kr_offsetof(kr_queue, data) + capacity);
    //kr_zero_memory((u8*)queue + kr_offsetof(kr_queue, data) + capacity / 2, capacity / 2);
  }

  queue->capacity = capacity;
  queue           = (kr_queue*)((u8*)queue + kr_offsetof(kr_queue, data));
  *address = queue;
}
