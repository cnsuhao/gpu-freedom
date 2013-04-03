unit serverjobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE for our jobs submitted
     to the server.

     All changes in the status column go through this class and its counterpart
     clientjobqueueworkflows.pas

     Workflow transitions on TBJOBQUEUE.status are documented in
     docs/dev/server-jobqueue-workflow-client.png

     (c) 2013 HB9TVM and the Global Processing Unit Team
}
interface

uses SyncObjs, SysUtils,
     dbtablemanagers, jobqueuetables, jobqueuehistorytables, workflowancestors,
     loggers;

// workflow for jobs that we process for someone else
type TServerJobQueueWorkflow = class(TJobQueueWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

end;

implementation

constructor TServerJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
end;



end.
