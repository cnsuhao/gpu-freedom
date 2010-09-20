unit computationthreads
{
    ComputationThread is a thread which executes stored in
    job.job.
}
interface

uses jobs, methodcontrollers, pluginmanager;

type
  TComputationThread = class(TThread)
   public
      
      constructor Create(var plugman : TPluginManager; var meth : TMethodController; var job : TJob; threadId : Longint);
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
	  plugMan_        :  TPluginManager;
      methController_ :  TMethodController;
      // if the thread is finished, job done is set to finish
      jobDone_: boolean;
   end;	  

end;

implementation

constructor TComputationThread.Create(var plugman : TPluginManager; var meth : TMethodController; var job : TJob; threadId : Longint);
begin
  inherited Create();
  
  plugMan_ := plugman;
  methController_ := meth;
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
 parser := TGPUParser.Create(plugman_, methController_, job_, thrdId_);
 parser.parse();
 parser.Free;
 syncOnJobFinished;
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