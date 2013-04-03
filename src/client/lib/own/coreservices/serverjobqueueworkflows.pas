unit serverjobqueueworkflows;
{
     This class implements the workflow for the table TBJOBQUEUE for our jobs submitted
     to the server. The server submits these jobs to other clients, which do the
     processing for us.

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

// workflow for jobs that other clients process for us
type TServerJobQueueWorkflow = class(TJobQueueWorkflowAncestor)
       constructor Create(var tableman : TDbTableManager; var logger : TLogger);

       // entry points for services
       function findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForWUUpload(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForJobUpload(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForStatusRetrieval(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForWURetrieval(var row : TDbJobQueueRow) : Boolean;
       function findRowInStatusForResultRetrieval(var row : TDbJobQueueRow) : Boolean;

       // standard workflow
       function changeStatusFromNewToForWUUpload(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForWUUploadToUploadingWorkunit(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromUploadingWorkunitToForJobUpload(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForJobUploadToUploadingJob(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromUploadingJobToForStatusRetrieval(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForStatusRetrievalToRetrievingStatus(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRetrievingStatusToForWuRetrieval(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForWuRetrievalToRetrievingWU(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRetrievingWUToForResultRetrieval(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromForResultRetrievalToRetrievingResult(var row : TDbJobQueueRow) : Boolean;
       function changeStatusFromRetrievingResultToCompleted(var row : TDbJobQueueRow) : Boolean;


end;

implementation

constructor TServerJobQueueWorkflow.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create(tableman, logger);
end;

// ********************************
// entry points for services
// ********************************
function TServerJobQueueWorkflow.findRowInStatusNew(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, S_NEW);
end;

function TServerJobQueueWorkflow.findRowInStatusForWUUpload(var row : TDbJobQueueRow) : Boolean;
begin
  Result := findRowInStatus(row, S_FOR_WU_UPLOAD);
end;


function TServerJobQueueWorkflow.findRowInStatusForJobUpload(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, S_FOR_JOB_UPLOAD);
end;

function TServerJobQueueWorkflow.findRowInStatusForStatusRetrieval(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, S_FOR_STATUS_RETRIEVAL);
end;

function TServerJobQueueWorkflow.findRowInStatusForWURetrieval(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, S_FOR_WU_RETRIEVAL);
end;

function TServerJobQueueWorkflow.findRowInStatusForResultRetrieval(var row : TDbJobQueueRow) : Boolean;
begin
 Result := findRowInStatus(row, S_FOR_RESULT_RETRIEVAL);
end;


// ********************************
// standard workflow
// ********************************
function TServerJobQueueWorkflow.changeStatusFromNewToForWUUpload(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_NEW, S_FOR_WU_UPLOAD, '');
end;

function TServerJobQueueWorkflow.changeStatusFromForWUUploadToUploadingWorkunit(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_FOR_WU_UPLOAD, S_UPLOADING_WORKUNIT, '');
end;

function TServerJobQueueWorkflow.changeStatusFromUploadingWorkunitToForJobUpload(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_UPLOADING_WORKUNIT, S_FOR_JOB_UPLOAD, '');
end;

function TServerJobQueueWorkflow.changeStatusFromForJobUploadToUploadingJob(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_FOR_JOB_UPLOAD, S_UPLOADING_JOB, '');
end;

function TServerJobQueueWorkflow.changeStatusFromUploadingJobToForStatusRetrieval(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_UPLOADING_JOB, S_FOR_STATUS_RETRIEVAL, '');
end;

function TServerJobQueueWorkflow.changeStatusFromForStatusRetrievalToRetrievingStatus(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_FOR_STATUS_RETRIEVAL, S_RETRIEVING_STATUS, '');
end;

function TServerJobQueueWorkflow.changeStatusFromRetrievingStatusToForWuRetrieval(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_RETRIEVING_STATUS, S_FOR_WU_RETRIEVAL, '');
end;

function TServerJobQueueWorkflow.changeStatusFromForWuRetrievalToRetrievingWU(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_FOR_WU_RETRIEVAL, S_RETRIEVING_WU, '');
end;

function TServerJobQueueWorkflow.changeStatusFromRetrievingWUToForResultRetrieval(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_RETRIEVING_WU, S_FOR_RESULT_RETRIEVAL, '');
end;

function TServerJobQueueWorkflow.changeStatusFromForResultRetrievalToRetrievingResult(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_FOR_RESULT_RETRIEVAL, S_RETRIEVING_RESULT, '');
end;

function TServerJobQueueWorkflow.changeStatusFromRetrievingResultToCompleted(var row : TDbJobQueueRow) : Boolean;
begin
   Result := changeStatus(row, S_RETRIEVING_RESULT, S_COMPLETED, '');
end;



end.
