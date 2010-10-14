unit computationthreads;
{
    ComputationThread is a thread which executes stored in
    job.job.
    
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses  Classes,
      jobs, methodcontrollers, pluginmanagers, resultcollectors, frontendmanagers,
      jobparsers;

type
  TComputationThread = class(TThread)
   public
      
    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager;
                       var job : TJob; thrdId : Longint);
    function    isJobDone : Boolean;
	  
	  
   protected
    procedure Execute; override;
    
    procedure SyncOnJobCreated;
    procedure SyncOnJobFinished;
	  
   private
      job_done_      : Boolean;
      // input parameters
      // the job which needs to be computed
      job_           :  TJob;
	  // thread id for GPU component
      thrdID_        : Longint;
      // helper structures
      plugman_        : TPluginManager;
      methController_ : TMethodController;
      rescoll_        : TResultCollector;
      frontman_       : TFrontendManager;

      // if the thread is finished, job done is set to finish
      jobDone_: boolean;
   end;	  

implementation

constructor TComputationThread.Create(var plugman : TPluginManager; var meth : TMethodController;
                                      var res : TResultCollector; var frontman : TFrontendManager;
                                      var job : TJob; thrdId : Longint);
begin
  jobDone_ := false;
  inherited Create(false);


  job_ := job;
  thrdId_ := thrdId;

  plugman_ := plugman;
  methController_ := meth;
  rescoll_ := res;
  frontman_ := frontman;
end;

function  TComputationThread.isJobDone : Boolean;
begin
  Result := jobDone_;
end;

procedure  TComputationThread.Execute;
var parser : TJobParser;
begin
 syncOnJobCreated;
 parser := TJobParser.Create(plugman_, methController_, rescoll_, frontman_, job_, thrdId_);
 parser.parse();
 parser.Free;
 syncOnJobFinished;
 jobDone_ := true;
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
