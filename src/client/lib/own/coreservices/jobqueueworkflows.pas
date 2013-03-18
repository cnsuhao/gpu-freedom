unit jobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE.
     All changes in the status column go through this class.

     (c) 2013 HB9TVM and the Global Processing Unit Team
}
interface

uses SyncObjs,
     dbtablemanagers, jobqueuetables, workflowancestors, loggers;

type TJobQueueWorkflow = class(TWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

     private
       procedure changeStatus(row : TDbJobQueueRow; fromS, toS : Longint);
end;

implementation

constructor TJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
end;

procedure TJobQueueWorkflow.changeStatus(row : TDbJobQueueRow; fromS, toS : Longint);
begin
  CS_.Enter;
  // perform status change
  CS_.Leave;
end;


end.
