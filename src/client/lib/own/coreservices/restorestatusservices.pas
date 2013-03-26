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
  while workflowman_.getJobQueueWorkflow().findRowInStatusRetrievingWorkunitForRestoral(jobqueuerow_) do
        begin
          workflowman_.getJobQueueWorkflow().restoreFromRetrievingWorkunit(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status RETRIEVING_WORKUNIT set to FOR_WU_RETRIEVAL');
        end;

  while workflowman_.getJobQueueWorkflow().findRowInStatusAcknowledgingForRestoral(jobqueuerow_) do
        begin
          workflowman_.getJobQueueWorkflow().restoreFromAcknowledging(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status ACKNOWLEDGING set to FOR_ACKNOWLEDGEMENT');
        end;

  while workflowman_.getJobQueueWorkflow().findRowInStatusRunningForRestoral(jobqueuerow_) do
        begin
          workflowman_.getJobQueueWorkflow().restoreFromRunning(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status RUNNING set to READY');
        end;

  while workflowman_.getJobQueueWorkflow().findRowInStatusTransmittingWorkunitForRestoral(jobqueuerow_) do
        begin
          workflowman_.getJobQueueWorkflow().restoreFromTransmittingWorkunit(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status TRANSMITTING_WORKUNIT set to FOR_WU_TRANSMISSION');
        end;

  while workflowman_.getJobQueueWorkflow().findRowInStatusTransmittingResultForRestoral(jobqueuerow_) do
        begin
          workflowman_.getJobQueueWorkflow().restoreFromTransmittingResult(jobqueuerow_, logHeader_+' Restored at bootstrap');
          logger_.log(LVL_DEBUG, logHeader_+'Found job in status TRANSMITTING_RESULT set to FOR_RESULT_TRANSMISSION');
        end;

  logger_.log(LVL_INFO, logHeader_+'Restore transitional status service  over.');
  done_ := True;
  erroneous_ := false;
end;


end.

