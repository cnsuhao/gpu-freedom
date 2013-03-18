unit workflowmanagers;
{
   TWorkflowManager handles all workflows inside Allegro
   (c) by 2010-2013 HB9TVM and the GPU Team
}

interface

uses dbtablemanagers, jobqueueworkflows, loggers;


type TWorkflowManager = class(TObject)
    public
      constructor Create(var tableman : TDbTableManager; var logger : TLogger);
      destructor  Destroy;

      function getJobQueueWorkflow : TJobQueueWorkflow;

     private
      jobqueueworkflow_ : TJobQueueWorkflow;
end;

implementation

constructor TWorkflowManager.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create;
  jobqueueworkflow_ := TJobQueueWorkflow.Create(tableman, logger);
end;

destructor  TWorkflowManager.Destroy;
begin
  jobqueueworkflow_.Free;
  inherited Destroy;
end;

function TWorkflowManager.getJobQueueWorkflow : TJobQueueWorkflow;
begin
  Result := jobqueueworkflow_;
end;

end.
