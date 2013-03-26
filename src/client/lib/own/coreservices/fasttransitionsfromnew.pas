unit fasttransitionsfromnew;

interface

uses
  coreservices, loggers, configurations, dbtablemanagers, workflowmanagers,
  Classes, SysUtils;

type TFastTransitionFromNewServiceThread = class(TCoreServiceThread)
 public
   constructor Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                      var workflowman : TWorkflowManager);

  protected
    workflowman_ : TWorkflowManager;
    jobqueuerow_ : TDbJobQueueRow;

    procedure Execute; override;

  private
    procedure applyWorkflow;

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
  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from new started...');
  while workflowman_.getJobQueueWorkflow().findRowInStatusNew(jobqueuerow_) do
        ApplyWorkflow;

  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from new over...');
  done_ := True;
  erroneous := false;
end;

procedure TFastTransitionFromNewServiceThread.applyWorkflow;
begin
  if jobqueuerow_.islocal then
     begin
       if (Trim(jobqueuerow_.workunitjobpath)<>'') and (not FileExists(jobqueuerow_.workunitjobpath)) then
          workflowman_.getJobQueueWorkflow().changeStatusToError(jobqueuerow_, 'Job is local, but workunit job does not exist on filesystem ('+jobqueuerow_.workunitjobpath+')')
       else
          workflowman_.getJobQueueWorkflow().changeStatusFromNewToReady(jobqueuerow_, logHeader+'Fast transition as job is local and workunit check is ok');
     end
  else
     begin
        // this is a global job
        if Trim(dbqueuerow.workunitjob)='' then
              begin
                if dbqueuerow.requireack then
                   workflowman_.getJobQueueWorkflow().changeStatusFromNewToForAcknowledgement(dbqueuerow, logHeader_+'Fast transition: no workunit to be retrieved but ack required');
                else
                   workflowman_.getJobQueueWorkflow().changeStatusFromNewToReady(dbqueuerow, logHeader_+'Fast transition: jobqueue does not require acknowledgement and does not have workunits.');
              end
        else
           // standard workflow
           workflowman_.getJobQueueWorkflow().changeStatusFromNewToForWURetrieval(jobqueuerow_);
     end;

end;

end.

