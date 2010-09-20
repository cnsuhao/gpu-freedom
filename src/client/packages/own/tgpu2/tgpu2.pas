{
  TGPU2 class is the main class which is the core of the core of GPU itself.

  It  keeps track of MAX_THREADS slots which can contain a running
  ComputationThread. A new ComputationThread can be created on a slot
  by using the Compute(...) method after defining a TJob structure. 
  This class is the only class which can istantiate a new ComputationThread.
  
  Results are collected into the JobsCollected array. 
  
  TGPU2 component encapsulates the PluginManager which manages Plugins
  which contain the computational algorithms.
  
  It encapsulates the MethodController, which makes sure that the same function
  inside a plugin is not called concurrently, for increased stability.
    
}
unit TGPU2;

interface

uses
  jobs, computationthreads, stacks, tgpuconstants;

type
  TGPUCore2 = class(TObject)
  public

    constructor Create(var plugman : TPluginManager; var meth : TMethodController);
    destructor  Destroy();
 
    // the number of threads can be changed dynamically
    procedure setMaxThreads(x: Longint);
    function  getMaxThreads() : Longint;
    function  isIdle() : Boolean;
    function  getCurrentThreads : Longint;
    
  private
    max_threads_, 
	   threads_running : Longint;
    is_idle_        : Boolean;
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    
    threads_        : Array[1..MAX_THREADS] of TComputationThread;
  end;
  
implementation

constructor TGPUCore2.Create(var plugman : TPluginManager; var meth : TMethodController);
begin
  inherited Create();
  plugman_ := plugman;
  methController_ := meth;
  max_threads_ := DEFAULT_THREADS;
end;

destructor TGPUCore2.Destroy();
begin
  inherited;
end;

procedure TGPUCore2.setMaxThreads(x: Longint);
begin
  max_threads_ := x;
end;

function  TGPUCore2.getMaxThreads() : Longint;
begin
 Result := max_threads_;
end;

function  TGPUCore2.isIdle() : Boolean;
begin
 Result := (current_threas_ = 0);
end;

function  TGPUCore2.getCurrentThreads : Longint;
begin
 Result := current_threads_;
end;


end;
  