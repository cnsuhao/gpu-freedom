unit jobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE.
     All changes in the status column go through this class.

     (c) 2013 HB9TVM and the Global Processing Unit Team
}
interface

uses SyncObjs, SysUtils,
     dbtablemanagers, jobqueuetables, jobqueuehistorytables, workflowancestors,
     loggers;

type TJobQueueWorkflow = class(TWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

       function findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;

       function changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromComputedToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusToError(var row : TDbJobQueueRow; errormsg : String) : Boolean;

     private
       function findRowInStatus(var row : TDbJobQueueRow; s : TJobStatus) : Boolean;
       function changeStatus(var row : TDbJobQueueRow; fromS, toS : TJobStatus; message : String) : Boolean;
end;

implementation

constructor TJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
end;

function TJobQueueWorkflow.findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_NEW);
end;

function TJobQueueWorkflow.findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, JS_READY);
end;

function TJobQueueWorkflow.findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_COMPLETED);
end;

function TJobQueueWorkflow.findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_COMPUTED);
end;

function TJobQueueWorkflow.changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_NEW, JS_READY, '');
end;

function TJobQueueWorkflow.changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_READY, JS_RUNNING, '');
end;

function TJobQueueWorkflow.changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_RUNNING, JS_COMPUTED, '');
end;

function TJobQueueWorkflow.changeStatusFromComputedToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_COMPUTED, JS_COMPLETED, '');
end;


function TJobQueueWorkflow.changeStatusToError(var row : TDbJobQueueRow; errormsg : String) : Boolean;
begin
  Result := changeStatus(row, row.status, JS_ERROR, errormsg);
end;

function TJobQueueWorkflow.changeStatus(var row : TDbJobQueueRow; fromS, toS : TJobStatus; message : String) : Boolean;
var
   dbqueuehistoryrow : TDbJobQueueHistoryRow;
begin
  Result := false;
  if row.status<>fromS then
        begin
          logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   JobQueueStatusToString(fromS)+' because row has status '+JobQueueStatusToString(row.status));
          Exit;
        end;

  CS_.Enter;
  try
    row.status := toS;
    row.update_dt := Now;

    dbqueuehistoryrow.status     := row.status;
    dbqueuehistoryrow.jobqueueid := row.jobqueueid;
    dbqueuehistoryrow.message    := message;

    tableman_.getJobQueueTable().insertOrUpdate(row);
    tableman_.getJobQueueHistoryTable().insert(dbqueuehistoryrow);
    Result := true;
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   JobQueueStatusToString(fromS)+' because of Exception '+e.ToString);
  end;

  CS_.Leave;
  if Result then logger_.log(LVL_DEBUG, 'Jobqueue row '+row.jobdefinitionid+' changed status from '+
                                         JobQueueStatusToString(fromS)+' to status '+JobQueueStatusToString(row.status));
end;

function TJobQueueWorkflow.findRowInStatus(var row : TDbJobQueueRow; s : TJobStatus) : Boolean;
begin
  Result := false;
  CS_.Enter;
  try
     Result := tableman_.getJobQueueTable().findRowInStatus(row, s);
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to retrieve row in status '+
                                   JobQueueStatusToString(s)+' because of Exception '+e.ToString);
  end;

  CS_.Leave;
end;

end.
