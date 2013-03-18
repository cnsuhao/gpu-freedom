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

       function changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRunningToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromCompletedToTransmitted(var row : TDbJobQueueRow) : Boolean;

     private
       function changeStatus(row : TDbJobQueueRow; fromS, toS : Longint) : Boolean;
end;

implementation

constructor TJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
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

function TJobQueueWorkflow.changeStatus(row : TDbJobQueueRow; fromS, toS : Longint) : Boolean;
begin
  Result := false;
  if row.status<>fromS then
        begin
          logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   IntToStr(fromS)+' because row has status '+IntToStr(row.status));
          Exit;
        end;

  CS_.Enter;
  row.status := toS;
  tableman_.getJobQueueTable().insertOrUpdate(row);
  Result := true;
  CS_.Leave;

  logger_.log(LVL_DEBUG, 'Jobqueue row '+row.jobdefinitionid+' changed status from '+
              IntToStr(fromS)+' to status '+IntToStr(row.status));
end;


end.
