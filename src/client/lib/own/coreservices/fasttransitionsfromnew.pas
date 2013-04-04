unit fasttransitionsfromnew;

interface

uses
  coreservices, loggers, coreconfigurations, dbtablemanagers, workflowmanagers,
  jobqueuetables, Classes, SysUtils;

type TFastTransitionFromNewServiceThread = class(TCoreServiceThread)
 public
   constructor Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                      var workflowman : TWorkflowManager);

  protected
    workflowman_ : TWorkflowManager;
    jobqueuerow_ : TDbJobQueueRow;

    procedure Execute; override;

  private
    procedure applyClientWorkflow;
    procedure applyServerWorkflow;
end;

implementation

constructor TFastTransitionFromNewServiceThread.Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                                                       var workflowman : TWorkflowManager);
begin
  inherited Create(logger, logHeader, conf, tableman);
  workflowman_ := workflowman;
end;


procedure TFastTransitionFromNewServiceThread.Execute;
begin
  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from NEW started...');
  while workflowman_.getClientJobQueueWorkflow().findRowInStatusNew(jobqueuerow_) do
        ApplyClientWorkflow;

  while workflowman_.getServerJobQueueWorkflow().findRowInStatusNew(jobqueuerow_) do
        ApplyServerWorkflow;

  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from NEW over...');
  done_ := True;
  erroneous_ := false;
end;

procedure TFastTransitionFromNewServiceThread.applyClientWorkflow;
begin
  if jobqueuerow_.islocal then
     begin
       if (Trim(jobqueuerow_.workunitjobpath)<>'') and (not FileExists(jobqueuerow_.workunitjobpath)) then
          workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Job is local, but workunit job does not exist on filesystem ('+jobqueuerow_.workunitjobpath+')')
       else
          workflowman_.getClientJobQueueWorkflow().changeStatusFromNewToReady(jobqueuerow_, logHeader_+'Fast transition as job is local and workunit check is ok');
     end
  else
     begin
        // this is a global job
        if Trim(jobqueuerow_.workunitjob)='' then
              begin
                if jobqueuerow_.requireack then
                   workflowman_.getClientJobQueueWorkflow().changeStatusFromNewToForAcknowledgement(jobqueuerow_, logHeader_+'Fast transition: no workunit to be retrieved but ack required')
                else
                   workflowman_.getClientJobQueueWorkflow().changeStatusFromNewToReady(jobqueuerow_, logHeader_+'Fast transition: jobqueue does not require acknowledgement and does not have workunits to be retrieved.');
              end
        else
           // standard workflow
           workflowman_.getClientJobQueueWorkflow().changeStatusFromNewToForWURetrieval(jobqueuerow_);
     end;

end;

procedure TFastTransitionFromNewServiceThread.applyServerWorkflow;
begin
   if (Trim(jobqueuerow_.workunitjobpath)='') then
         workflowman_.getServerJobQueueWorkflow().changeStatusFromNewToForJobUpload(jobqueuerow_, logHeader_+'Fast transition, no workunit job to be uploaded')
   else
   if (Trim(jobqueuerow_.workunitjobpath)<>'') and (not FileExists(jobqueuerow_.workunitjobpath)) then
         workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Workunit job does not exist on filesystem ('+jobqueuerow_.workunitjobpath+')')
   else
         workflowman_.getServerJobQueueWorkflow().changeStatusFromNewToForWUUpload(jobqueuerow_);
end;

end.

