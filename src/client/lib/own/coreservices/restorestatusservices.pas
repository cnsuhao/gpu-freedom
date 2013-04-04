unit restorestatusservices;

interface

uses
  coreservices, loggers, coreconfigurations, dbtablemanagers, workflowmanagers,
  jobqueuetables, Classes, SysUtils;

type TRestoreStatusServiceThread = class(TCoreServiceThread)
 public
   constructor Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                      var workflowman : TWorkflowManager);

  protected
    workflowman_ : TWorkflowManager;
    jobqueuerow_ : TDbJobQueueRow;

    procedure Execute; override;

end;

implementation

constructor TRestoreStatusServiceThread.Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                                                       var workflowman : TWorkflowManager);
begin
  inherited Create(logger, logHeader, conf, tableman);
  workflowman_ := workflowman;
end;


procedure TRestoreStatusServiceThread.Execute;
begin
  logger_.log(LVL_INFO, logHeader_+'Restore transitional status service started...');
  //****************
  //* Client stati
  //****************
  while workflowman_.getClientJobQueueWorkflow().findRowInStatusRetrievingWorkunitForRestoral(jobqueuerow_) do
        begin
          workflowman_.getClientJobQueueWorkflow().restoreFromRetrievingWorkunit(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status C_RETRIEVING_WORKUNIT set to C_FOR_WU_RETRIEVAL');
        end;

  while workflowman_.getClientJobQueueWorkflow().findRowInStatusAcknowledgingForRestoral(jobqueuerow_) do
        begin
          workflowman_.getClientJobQueueWorkflow().restoreFromAcknowledging(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status C_ACKNOWLEDGING set to C_FOR_ACKNOWLEDGEMENT');
        end;

  while workflowman_.getClientJobQueueWorkflow().findRowInStatusRunningForRestoral(jobqueuerow_) do
        begin
          workflowman_.getClientJobQueueWorkflow().restoreFromRunning(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status C_RUNNING set to C_READY');
        end;

  while workflowman_.getClientJobQueueWorkflow().findRowInStatusTransmittingWorkunitForRestoral(jobqueuerow_) do
        begin
          workflowman_.getClientJobQueueWorkflow().restoreFromTransmittingWorkunit(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status C_TRANSMITTING_WORKUNIT set to C_FOR_WU_TRANSMISSION');
        end;

  while workflowman_.getClientJobQueueWorkflow().findRowInStatusTransmittingResultForRestoral(jobqueuerow_) do
        begin
          workflowman_.getClientJobQueueWorkflow().restoreFromTransmittingResult(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status C_TRANSMITTING_RESULT set to C_FOR_RESULT_TRANSMISSION');
        end;

  //****************
  //* Server stati
  //****************
  while workflowman_.getServerJobQueueWorkflow().findRowInStatusUploadingWorkunitForRestoral(jobqueuerow_) do
        begin
          workflowman_.getServerJobQueueWorkflow().restoreFromUploadingWorkunit(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status S_UPLOADING_WORKUNIT set to S_FOR_WU_UPLOAD');
        end;

  while workflowman_.getServerJobQueueWorkflow().findRowInStatusUploadingJobForRestoral(jobqueuerow_) do
        begin
          workflowman_.getServerJobQueueWorkflow().restoreFromUploadingJob(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status S_UPLOADING_JOB set to S_FOR_JOB_UPLOAD');
        end;

  while workflowman_.getServerJobQueueWorkflow().findRowInStatusRetrievingStatusForRestoral(jobqueuerow_) do
        begin
          workflowman_.getServerJobQueueWorkflow().restoreFromRetrievingStatus(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status S_RETRIEVING_STATUS set to S_FOR_STATUS_RETRIEVAL');
        end;

  while workflowman_.getServerJobQueueWorkflow().findRowInStatusRetrievingWUForRestoral(jobqueuerow_) do
        begin
          workflowman_.getServerJobQueueWorkflow().restoreFromRetrievingWU(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status S_RETRIEVING_WU set to S_FOR_WU_RETRIEVAL');
        end;

  while workflowman_.getServerJobQueueWorkflow().findRowInStatusRetrievingResultForRestoral(jobqueuerow_) do
        begin
          workflowman_.getServerJobQueueWorkflow().restoreFromRetrievingResult(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status S_RETRIEVING_RESULT set to S_FOR_RESULT_RETRIEVAL');
        end;


  logger_.log(LVL_INFO, logHeader_+'Restore transitional status service  over.');
  done_ := True;
  erroneous_ := false;
end;


end.

