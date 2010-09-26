unit computationthreads;
{
    ComputationThread is a thread which executes stored in
    job.job.
    
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses jobs, methodcontrollers, pluginmanager;

type
  TComputationThread = class(TThread)
   public
      
    constructor Create(var core : TGPU2Core; var job : TJob; threadId : Longint);
    function    isJobDone : Boolean;
	  
	  
   protected
    procedure Execute; override;
    
 	procedure SyncOnJobCreated;
    procedure SyncOnJobFinished;
	  
   private
      job_done_    : Boolean; 
      // input parameters
	  // the job which needs to be computed
      job_            :  TJob;
	  // thread id for GPU component
      thrdID_        : Longint;
 	  // helper structures
	  core_           : TGPU2Core
      // if the thread is finished, job done is set to finish
      jobDone_: boolean;
   end;	  

end;

implementation

constructor TComputationThread.Create(var core : TGPU2Core; var job : TJob; threadId : Longint);
begin
  inherited Create(true);
  
  jobDone_ := false;
  core_ := core;
  job_ := job;
  thrdId_ := threadId;
end;

function  TComputationThread.isJobDone : Boolean;
begin
  Result := jobDone_;
end;

procedure  TComputationThread.Execute; override;
var parser : TGPUParser;
begin
 syncOnJobCreated;
 parser := TGPUParser.Create(core_, job_, thrdId_);
 parser.parse();
 parser.Free;
 syncOnJobFinished;
 jobDone := true;
end;


procedure TComputationThread.SyncOnJobCreated;
begin
try
    if not Terminated and Assigned(job_.OnCreated) then
      job_.OnCreated(job_);
  except
  end;
end;

procedure TComputationThread.SyncOnJobFinished;
begin
  try //maybe a overkill, but make sure thread continues.
    if not Terminated and Assigned(job_.OnFinished) then
      job_.OnFinished(job_);
  except
  end;
end;


end.
