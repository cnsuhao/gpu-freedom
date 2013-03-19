unit jobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE.
     All changes in the status column go through this class.

     (c) 2013 HB9TVM and the Global Processing Unit Team
}
interface

uses SyncObjs, SysUtils,
     dbtablemanagers, jobqueuetables, workflowancestors, loggers;

type TJobQueueWorkflow = class(TWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

       function findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusTransmitted(var row : TDbJobQueueRow) : Boolean;

       function changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRunningToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromCompletedToTransmitted(var row : TDbJobQueueRow) : Boolean;

     private
       function findRowInStatus(row : TDbJobQueueRow; s : TJobStatus) : Boolean;
       function changeStatus(row : TDbJobQueueRow; fromS, toS : TJobStatus) : Boolean;
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

function TJobQueueWorkflow.findRowInStatusTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_TRANSMITTED);
end;

function TJobQueueWorkflow.changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
begin
  if (not row.requireack) then
     begin
       logger_.log(LVL_SEVERE, 'Internal error: a job which does not require acknowledgmente should not do a transition from NEW to READY ('+row.jobqueueid+')');
       Result := false;
       Exit;
     end;
  Result := changeStatus(row, JS_NEW, JS_READY);
end;

function TJobQueueWorkflow.changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_READY, JS_RUNNING);
end;

function TJobQueueWorkflow.changeStatusFromRunningToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_RUNNING, JS_COMPLETED);
end;

function TJobQueueWorkflow.changeStatusFromCompletedToTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  if (row.islocal) then
     begin
       logger_.log(LVL_SEVERE, 'Internal error: a local job should not do a transition from COMPLETED to TRANSMITTED ('+row.jobqueueid+')');
       Result := false;
       Exit;
     end;

  Result := changeStatus(row, JS_COMPLETED, JS_TRANSMITTED);
end;

function TJobQueueWorkflow.changeStatus(row : TDbJobQueueRow; fromS, toS : TJobStatus) : Boolean;
begin
  Result := false;
  if row.status<>fromS then
        begin
          logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   IntToStr(fromS)+' because row has status '+IntToStr(row.status));
          Exit;
        end;

  CS_.Enter;
  try
    row.status := toS;
    tableman_.getJobQueueTable().insertOrUpdate(row);
    Result := true;
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   IntToStr(fromS)+' because of Exception '+e.ToString);
  end;

  CS_.Leave;
  if Result then logger_.log(LVL_DEBUG, 'Jobqueue row '+row.jobdefinitionid+' changed status from '+
                                         IntToStr(fromS)+' to status '+IntToStr(row.status));
end;

function TJobQueueWorkflow.findRowInStatus(row : TDbJobQueueRow; s : TJobStatus) : Boolean;
begin
  Result := false;
  CS_.Enter;
  try
     Result := tableman_.getJobQueueTable().findRowInStatus(row, s);
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to retrieve row in status '+
                                   IntToStr(s)+' because of Exception '+e.ToString);
  end;

  CS_.Leave;
end;

end.
