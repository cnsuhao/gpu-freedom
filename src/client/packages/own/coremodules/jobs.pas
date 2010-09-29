unit jobs;
{
  In order to create a new ComputationThread via TGPU::Compute()
  one needs first to define a TJob structure which encapsulates
  all information needed to perform a task.
  
  StackCommands contain a sequence of comma separated values
  and function calls. StackCommands might contain for example
  1,1,add   
  'g','pu',concat
  (see also the stack description at
  http://gpu.sourceforge.net/virtual.php). 

  JobId is a client unique identifier for the job.
  JobSlot is used by the TGPU_component class, to remember
  which ComputationThread is computing this TJob.
  Stack is an internal structure which can be passed to plugins.
  
  ComputedTime stores first the start time of the computation,
  then the total amount of time (wall time), used to finish the job.
  
  OnCreated, OnProgress, OnFinish, OnError are events to which
  TGPUTella in gpu_main_form.pas listens. TGPUTella is the main
  form window and these events help to keep the job list updated.
  
   (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
  
}
interface

uses Classes, stacks;

type
  TNotifyEvent = procedure(Sender: TObject) of object;

type
  TJob = class(TObject)
  public
    OnCreated, OnProgress, OnFinished, OnError: TNotifyEvent;

    JobID:   string;
    JobSlot: integer;

    stack: TStack;
    Job,                 // this contains the job itself
    JobResult : string;  // this contains the result of the computation
    
    hasError : Boolean;  // there is an error in the Job structure,
                         // on stack.error

    ComputedTime: TDateTime;
  end;

type
  TJobOnGnutella = class(TJob)
    JobGUID: TGUID;

    IncomingOrder: integer;
  end;

implementation

end.
