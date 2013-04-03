unit clientjobqueueworkflows;
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

// workflow for jobs that we process for someone else
type TClientJobQueueWorkflow = class(TJobQueueWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

       // entry points for services
       function findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForWURetrieval(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForAcknowledgement(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForWUTransmission(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForResultTransmission(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;


       // standard workflow
       function changeStatusFromNewToForWURetrieval(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForWURetrievalToRetrievingWorkunit(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRetrievingWorkunitToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromWorkunitRetrievedToForAcknowledgement(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForAcknowledgementToAcknowledging(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromAcknowledgingToReady(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromComputedToForWUTransmission(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForWUTransmissionToTransmittingWorkunit(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromTransmittingWorkunitToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromWorkunitTransmittedToForResultTransmission(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForResultTransmissionToTransmittingResult(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromTransmittingResultToCompleted(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromCompletedToWorkunitsCleanedUp(var row : TDbJobQueueRow) : Boolean;

       // transition shortcuts
       function changeStatusFromNewToReady(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function changeStatusFromComputedToCompleted(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function changeStatusFromNewToForWURetrieval(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function changeStatusFromNewToForAcknowledgement(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function changeStatusFromWorkunitRetrievedToReady(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function changeStatusFromComputedToForResultTransmission(var row : TDbJobQueueRow; msgdesc : String) : Boolean;

       // restore transitions
       function findRowInStatusRetrievingWorkunitForRestoral(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusAcknowledgingForRestoral(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusRunningForRestoral(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusTransmittingWorkunitForRestoral(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusTransmittingResultForRestoral(var row : TDbJobQueueRow) : Boolean;

       function restoreFromRetrievingWorkunit(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function restoreFromAcknowledging(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function restoreFromRunning(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function restoreFromTransmittingWorkunit(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
       function restoreFromTransmittingResult(var row : TDbJobQueueRow; msgdesc : String) : Boolean;

       // error transition
       function changeStatusToError(var row : TDbJobQueueRow; errormsg : String) : Boolean;
end;

implementation

constructor TClientJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
end;

// ********************************
// entry points for services
// ********************************
function TClientJobQueueWorkflow.findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_NEW);
end;

function TClientJobQueueWorkflow.findRowInStatusForWURetrieval(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_FOR_WU_RETRIEVAL);
end;

function TClientJobQueueWorkflow.findRowInStatusForAcknowledgement(var row : TDbJobQueueRow) : Boolean;
begin
   Result := findRowInStatus(row, C_FOR_ACKNOWLEDGEMENT);
end;

function TClientJobQueueWorkflow.findRowInStatusReady(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, C_READY);
end;

function TClientJobQueueWorkflow.findRowInStatusComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_COMPUTED);
end;

function TClientJobQueueWorkflow.findRowInStatusForWUTransmission(var row : TDbJobQueueRow) : Boolean;
begin
   Result := findRowInStatus(row, C_FOR_WU_TRANSMISSION);
end;

function TClientJobQueueWorkflow.findRowInStatusForResultTransmission(var row : TDbJobQueueRow) : Boolean;
begin
     Result := findRowInStatus(row, C_FOR_RESULT_TRANSMISSION);
end;

function TClientJobQueueWorkflow.findRowInStatusCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_COMPLETED);
end;

// ********************************
// * standard workflow
// ********************************
function TClientJobQueueWorkflow.changeStatusFromNewToForWURetrieval(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, C_NEW, C_FOR_WU_RETRIEVAL, '');
end;

function TClientJobQueueWorkflow.changeStatusFromForWURetrievalToRetrievingWorkunit(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_FOR_WU_RETRIEVAL, C_RETRIEVING_WORKUNIT, '');
end;

function TClientJobQueueWorkflow.changeStatusFromRetrievingWorkunitToWorkunitRetrieved(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_RETRIEVING_WORKUNIT, C_WORKUNIT_RETRIEVED, '');
end;

function TClientJobQueueWorkflow.changeStatusFromWorkunitRetrievedToForAcknowledgement(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_WORKUNIT_RETRIEVED, C_FOR_ACKNOWLEDGEMENT, '');
end;

function TClientJobQueueWorkflow.changeStatusFromForAcknowledgementToAcknowledging(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_FOR_ACKNOWLEDGEMENT, C_ACKNOWLEDGING, '');
end;

function TClientJobQueueWorkflow.changeStatusFromAcknowledgingToReady(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_ACKNOWLEDGING, C_READY, '');
end;

function TClientJobQueueWorkflow.changeStatusFromReadyToRunning(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_READY, C_RUNNING, '');
end;

function TClientJobQueueWorkflow.changeStatusFromRunningToComputed(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_RUNNING, C_COMPUTED, '');
end;

function TClientJobQueueWorkflow.changeStatusFromComputedToForWUTransmission(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_COMPUTED, C_FOR_WU_TRANSMISSION, '');
end;

function TClientJobQueueWorkflow.changeStatusFromForWUTransmissionToTransmittingWorkunit(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_FOR_WU_TRANSMISSION, C_TRANSMITTING_WORKUNIT, '');
end;

function TClientJobQueueWorkflow.changeStatusFromTransmittingWorkunitToWorkunitTransmitted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_TRANSMITTING_WORKUNIT, C_WORKUNIT_TRANSMITTED, '');
end;

function  TClientJobQueueWorkflow.changeStatusFromWorkunitTransmittedToForResultTransmission(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_WORKUNIT_TRANSMITTED, C_FOR_RESULT_TRANSMISSION, '');
end;

function  TClientJobQueueWorkflow.changeStatusFromForResultTransmissionToTransmittingResult(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_FOR_RESULT_TRANSMISSION, C_TRANSMITTING_RESULT, '');
end;

function TClientJobQueueWorkflow.changeStatusFromTransmittingResultToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_TRANSMITTING_RESULT, C_COMPLETED, '');
end;

function TClientJobQueueWorkflow.changeStatusFromCompletedToWorkunitsCleanedUp(var row : TDbJobQueueRow) : Boolean;
begin
  Result := changeStatus(row, C_COMPLETED, C_WORKUNITS_CLEANEDUP, '');
end;

// ********************************
// * transition shortcuts
// ********************************
function TClientJobQueueWorkflow.changeStatusFromNewToReady(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_NEW, C_READY, msgdesc);
end;

function TClientJobQueueWorkflow.changeStatusFromComputedToCompleted(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_COMPUTED, C_COMPLETED, msgdesc);
end;

function TClientJobQueueWorkflow.changeStatusFromNewToForWURetrieval(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_NEW, C_FOR_WU_RETRIEVAL, msgdesc);
end;

function TClientJobQueueWorkflow.changeStatusFromNewToForAcknowledgement(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_NEW, C_FOR_ACKNOWLEDGEMENT, msgdesc);
end;


function TClientJobQueueWorkflow.changeStatusFromWorkunitRetrievedToReady(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_WORKUNIT_RETRIEVED, C_READY, msgdesc);
end;

function TClientJobQueueWorkflow.changeStatusFromComputedToForResultTransmission(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_COMPUTED, C_FOR_RESULT_TRANSMISSION, msgdesc);
end;


// ********************************
// * error transition
// ********************************

function TClientJobQueueWorkflow.changeStatusToError(var row : TDbJobQueueRow; errormsg : String) : Boolean;
begin
  Result := changeStatus(row, row.status, C_ERROR, errormsg);
end;

// **********************************
// * restore from transitional status
// **********************************

function TClientJobQueueWorkflow.findRowInStatusRetrievingWorkunitForRestoral(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_RETRIEVING_WORKUNIT);
end;

function TClientJobQueueWorkflow.findRowInStatusAcknowledgingForRestoral(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_ACKNOWLEDGING);
end;

function TClientJobQueueWorkflow.findRowInStatusRunningForRestoral(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_RUNNING);
end;

function TClientJobQueueWorkflow.findRowInStatusTransmittingWorkunitForRestoral(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_TRANSMITTING_WORKUNIT);
end;

function TClientJobQueueWorkflow.findRowInStatusTransmittingResultForRestoral(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, C_TRANSMITTING_RESULT);
end;

function TClientJobQueueWorkflow.restoreFromRetrievingWorkunit(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_RETRIEVING_WORKUNIT, C_FOR_WU_RETRIEVAL, msgdesc);
end;

function TClientJobQueueWorkflow.restoreFromAcknowledging(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_ACKNOWLEDGING, C_FOR_ACKNOWLEDGEMENT, msgdesc);
end;

function TClientJobQueueWorkflow.restoreFromRunning(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_RUNNING, C_READY, msgdesc);
end;

function TClientJobQueueWorkflow.restoreFromTransmittingWorkunit(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_TRANSMITTING_WORKUNIT, C_FOR_WU_TRANSMISSION, msgdesc);
end;

function TClientJobQueueWorkflow.restoreFromTransmittingResult(var row : TDbJobQueueRow; msgdesc : String) : Boolean;
begin
  Result := changeStatus(row, C_TRANSMITTING_RESULT, C_FOR_RESULT_TRANSMISSION, msgdesc);
end;


end.
