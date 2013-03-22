unit jobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE.
     All changes in the status column go through this class.

     Workflow transitions on TBJOBQUEUE.status are documented in
     docs/dev/jobqueue-workflow-client.png

     (c) 2013 HB9TVM and the Global Processing Unit Team
}
interface

uses SyncObjs, SysUtils,
     dbtablemanagers, jobqueuetables, jobqueuehistorytables, workflowancestors,
     loggers;

type TJobQueueWorkflow = class(TWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

       // entry points for services
       function findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;

       // standard workflow
       function changeStatusFromNewToRetrievingWorkunit(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRetrievingWorkunitToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromWorkunitRetrievedToAcknowledging(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromAcknowledgingToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromComputedToTransmittingWorkunit(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromTransmittingWorkunitToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromWorkunitTransmittedToTransmittingResult(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromTransmittingResultToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromCompletedToWorkunitsCleanedUp(var row : TDbJobQueueRow) : Boolean;

       // transition shortcuts
       function changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromComputedToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromNewToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromWorkunitRetrievedToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromComputedToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;

       // error transition
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

// ********************************
// entry points for services
// ********************************
function TJobQueueWorkflow.findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_NEW);
end;

function TJobQueueWorkflow.findRowInStatusWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_WORKUNIT_RETRIEVED);
end;

function TJobQueueWorkflow.findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, JS_READY);
end;

function TJobQueueWorkflow.findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_COMPUTED);
end;

function TJobQueueWorkflow.findRowInStatusWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_WORKUNIT_TRANSMITTED);
end;

function TJobQueueWorkflow.findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, JS_COMPLETED);
end;

// ********************************
// * standard workflow
// ********************************
function TJobQueueWorkflow.changeStatusFromNewToRetrievingWorkunit(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_NEW, JS_RETRIEVING_WORKUNIT, '');
end;

function TJobQueueWorkflow.changeStatusFromRetrievingWorkunitToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_RETRIEVING_WORKUNIT, JS_WORKUNIT_RETRIEVED, '');
end;


function TJobQueueWorkflow.changeStatusFromWorkunitRetrievedToAcknowledging(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_WORKUNIT_RETRIEVED, JS_ACKNOWLEDGING, '');
end;

function TJobQueueWorkflow.changeStatusFromAcknowledgingToReady(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_ACKNOWLEDGING, JS_READY, '');
end;

function TJobQueueWorkflow.changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_READY, JS_RUNNING, '');
end;

function TJobQueueWorkflow.changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_RUNNING, JS_COMPUTED, '');
end;

function TJobQueueWorkflow.changeStatusFromComputedToTransmittingWorkunit(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_COMPUTED, JS_TRANSMITTING_WORKUNIT , '');
end;

function TJobQueueWorkflow.changeStatusFromTransmittingWorkunitToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_TRANSMITTING_WORKUNIT, JS_WORKUNIT_TRANSMITTED, '');
end;

function TJobQueueWorkflow.changeStatusFromWorkunitTransmittedToTransmittingResult(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_WORKUNIT_TRANSMITTED, JS_TRANSMITTING_RESULT, '');
end;

function TJobQueueWorkflow.changeStatusFromTransmittingResultToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_TRANSMITTING_RESULT, JS_COMPLETED, '');
end;

function TJobQueueWorkflow.changeStatusFromCompletedToWorkunitsCleanedUp(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_COMPLETED, JS_WORKUNITS_CLEANEDUP, '');
end;

// ********************************
// * transition shortcuts
// ********************************
function TJobQueueWorkflow.changeStatusFromNewToReady(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_NEW, JS_READY, '');
end;

function TJobQueueWorkflow.changeStatusFromComputedToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_COMPUTED, JS_COMPLETED, '');
end;

function TJobQueueWorkflow.changeStatusFromNewToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_NEW, JS_WORKUNIT_RETRIEVED, '');
end;


function TJobQueueWorkflow.changeStatusFromWorkunitRetrievedToReady(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_WORKUNIT_RETRIEVED, JS_READY, '');
end;

function TJobQueueWorkflow.changeStatusFromComputedToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, JS_COMPUTED, JS_WORKUNIT_TRANSMITTED, '');
end;

// ********************************
// * error status
// ********************************

function TJobQueueWorkflow.changeStatusToError(var row : TDbJobQueueRow; errormsg : String) : Boolean;
begin
  Result := changeStatus(row, row.status, JS_ERROR, errormsg);
end;

// ********************************
// * private, internal methods
// ********************************
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
