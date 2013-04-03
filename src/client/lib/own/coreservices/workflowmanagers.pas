unit workflowmanagers;
{
   TWorkflowManager handles all workflows inside Allegro
   (c) by 2010-2013 HB9TVM and the GPU Team
}

interface

uses dbtablemanagers, clientjobqueueworkflows, serverjobqueueworkflows,
     loggers;


type TWorkflowManager = class(TObject)
    public
      constructor Create(var tableman : TDbTableManager; var logger : TLogger);
      destructor  Destroy;

      function getClientJobQueueWorkflow : TClientJobQueueWorkflow;
      function getServerJobQueueWorkflow : TServerJobQueueWorkflow;

     private
      clientjobqueueworkflow_ : TClientJobQueueWorkflow;
      serverjobqueueworkflow_ : TServerJobQueueWorkflow;
end;

implementation

constructor TWorkflowManager.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create;
  clientjobqueueworkflow_ := TClientJobQueueWorkflow.Create(tableman, logger);
  serverjobqueueworkflow_ := TServerJobQueueWorkflow.Create(tableman, logger);
end;

destructor  TWorkflowManager.Destroy;
begin
  clientjobqueueworkflow_.Free;
  inherited Destroy;
end;

function TWorkflowManager.getClientJobQueueWorkflow : TClientJobQueueWorkflow;
begin
  Result := clientjobqueueworkflow_;
end;

function TWorkflowManager.getServerJobQueueWorkflow : TServerJobQueueWorkflow;
begin
  Result := serverjobqueueworkflow_;
end;


end.
