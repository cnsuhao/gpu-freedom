{
  TGPU2 class is the main class which registers the GPU component on 
  the Freepascal bar.

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
unit TGPU_component;

interface

uses
  SysUtils, Classes, PluginManager, ComputationThread, Jobs, Definitions,
  common, gpu_utils, FrontendManager, FunctionCallController;

const
  MAX_THREADS = 64;  // maximum number of allowed threads

type
  TGPU = class(TComponent)
  public
    procedure setMaxThreads(x: Longint);

  private
    max_threads_, 
	current_threads : Longint;
    is_idle_        : Boolean;
  
  end;
  
implementation


end;
  