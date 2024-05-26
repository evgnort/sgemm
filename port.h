#ifndef _PORT_H
#define _PORT_H

#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>

const inline int posix_memalign(void **ptr, size_t align, size_t size)
	{ *ptr = _aligned_malloc(size,align); return *ptr ? 0 : errno; }

#define posix_memalign_free _aligned_free

#define THREAD_ID_TYPE HANDLE
#define THREAD_RETURN_TYPE DWORD
#define THREAD_MODIFIERS WINAPI

typedef LPTHREAD_START_ROUTINE CThreadRoutine;

const inline void maximizePriority(void)
	{ SetPriorityClass(GetCurrentProcess(),REALTIME_PRIORITY_CLASS); }

#define SOFT_BARRIER _ReadWriteBarrier()

#else

#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>

#define posix_memalign_free free

const inline void maximizePriority(void)
	{ setpriority(PRIO_PROCESS, 0, -20); }

typedef void *(*CThreadRoutine)(void*);

#define THREAD_ID_TYPE pthread_t 
#define THREAD_RETURN_TYPE void *
#define THREAD_MODIFIERS

#define SOFT_BARRIER asm volatile("": : :"memory")

#define min(A,B) (((A) < (B)) ? (A) : (B))

#endif

#define THREAD_FUNC THREAD_RETURN_TYPE THREAD_MODIFIERS
#define THREAD_RETURN(A) return (THREAD_RETURN_TYPE)(A)

int64_t get_nanotime(void);
void set_thread_processor(int pnum);
int get_thread_processor(void);

THREAD_ID_TYPE start_thread(CThreadRoutine func,void *params,int pnum);
void wait_thread(THREAD_ID_TYPE thread);

#endif // _PORT_H

