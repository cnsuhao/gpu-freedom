unit fasttransitionsfromcomputed;

interface

uses
  coreservices, loggers, coreconfigurations, dbtablemanagers, workflowmanagers,
  jobqueuetables, Classes, SysUtils;

type TFastTransitionFromComputedServiceThread = class(TCoreServiceThread)
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

constructor TFastTransitionFromComputedServiceThread.Create(var logger : TLogger; logHeader : String; var conf : TCoreConfiguration; var tableman : TDbTableManager;
                                                       var workflowman : TWorkflowManager);
begin
  inherited Create(logger, logHeader, conf, tableman);
  workflowman_ := workflowman;
end;


procedure TFastTransitionFromComputedServiceThread.Execute;
begin
  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from COMPUTED started...');
  while workflowman_.getJobQueueWorkflow().findRowInStatusComputed(jobqueuerow_) do
        ApplyWorkflow;

  logger_.log(LVL_DEBUG, logHeader_+'Service fast transition from COMPUTED over...');
  done_ := True;
  erroneous_ := false;
end;

procedure TFastTransitionFromComputedServiceThread.applyWorkflow;
begin
  if jobqueuerow_.islocal then
     begin
       if (Trim(jobqueuerow_.workunitresultpath)<>'') and (not FileExists(jobqueuerow_.workunitresultpath)) then
          workflowman_.getJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Job is local, but workunit result does not exist on filesystem ('+jobqueuerow_.workunitresultpath+')')
       else
          workflowman_.getJobQueueWorkflow().changeStatusFromComputedToCompleted(jobqueuerow_, logHeader_+'Fast transition as job is local and workunit check is ok');
     end
  else
     begin
        // this is a global job
        if Trim(jobqueuerow_.workunitresult)='' then
              begin
                   workflowman_.getJobQueueWorkflow().changeStatusFromComputedToForResultTransmission(jobqueuerow_, logHeader_+'Fast transition: no resulting workunit to be uploaded.');
              end
        else
           // standard workflow
           workflowman_.getJobQueueWorkflow().changeStatusFromComputedToForWUTransmission(jobqueuerow_);
     end;

end;

end.

