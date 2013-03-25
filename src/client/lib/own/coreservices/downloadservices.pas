unit downloadservices;
{
  DownloadServiceThread is a db aware thread which retrieves a file based
  on a jobqueue it retrieves. It handles the transition from NEW to WORKUNIT_RETRIEVED,
  temporarily setting the status to RETRIEVING_WORKUNIT

  This file is build with testhttp as template
   which is part of the Synapse library
  available under /src/client/lib/ext/synapse.

  (c) by 2002-2013 the GPU Development Team
  This unit is released under GNU Public License (GPL)
}

interface

uses
  managedthreads, servermanagers, workflowmanagers, jobqueuetables,
  downloadutils, loggers, sysutils;


type TDownloadServiceThread = class(TManagedThread)
 public
   constructor Create(var srv : TServerManager; var workflowman : TWorkflowManager;
                      proxy, port : String; var logger : TLogger);

 protected
    procedure Execute; override;

 private
    url_,
    proxy_,
    port_,
    logHeader_   : String;
    srv_         : TServerManager;
    workflowman_ : TWorkflowManager;
    logger_      : TLogger;

    jobqueuerow_ : TDbJobQueueRow;
end;


implementation


constructor TDownloadServiceThread.Create(var srv : TServerManager; var workflowman : TWorkflowManager;
                                   proxy, port : String; var logger : TLogger);
begin
  inherited Create(true); // suspended

  srv_         := srv;
  workflowman_ := workflowman;
  logger_      := logger;
  proxy_       := proxy;
  port_        := port;

  logHeader_   := '[TDownloadServiceThread]> ';
end;


procedure TDownloadServiceThread.execute();
var AltFilename : String;
    index       : Longint;
begin
   if not workflowman_.getJobQueueWorkflow().findRowInStatusNew(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status NEW. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

    if (Trim(jobqueuerow_.workunitjob)='') and (not jobqueuerow_.islocal) then
        begin
          logger_.log(LVL_WARNING, logHeader_+'Concurrency problem: found a global job in status NEW with no workunitjob to be retrieved.');
          // note, receiveservicejobs.pas is charged with this transition
          done_      := True;
          erroneous_ := True;
          Exit;
        end
    else
    if jobqueuerow_.islocal then
        begin
          logger_.log(LVL_WARNING, logHeader_+'Concurrency problem: Found a local job in status NEW.');
          // note: the local job inserter is charged with transitions
          erroneous_ := True;
          done_      := True;
          Exit;
        end
    else
    begin
          // Here comes the main loop to retrieve a workunit
          workflowman_.getJobQueueWorkflow().changeStatusFromNewToRetrievingWorkunit(jobqueuerow_);

          {
          erroneous_ := not downloadToFile(url_, targetPath_, targetFile_,
                        proxy_, port_,
                        'DownloadServiceThread ['+targetFile_+']> ', logger_);
          }
          if not (erroneous_) then
          begin
              workflowman_.getJobQueueWorkflow().changeStatusFromRetrievingWorkunitToWorkunitRetrieved(jobqueuerow_);
              if not jobqueuerow_.requireack then
                  workflowman_.getJobQueueWorkflow().changeStatusFromWorkUnitRetrievedToReady(jobqueuerow_, logHeader_+'Fast transition: jobqueue does not require acknowledgement.');
          end
          else workflowman_.getJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Unable to retrieve workunit!')

    end;

  {
  if FileExists(targetPath_+targetFile_) then
  begin
    index := 2;
    repeat
      AltFileName := targetFile_ + '.' + IntToStr(index);
      inc(index);
    until not FileExists(targetPath_+AltFileName);
    logger_.log(LVL_WARNING, '"'+targetFile_+'" exists, writing to "'+AltFileName+'"');
    targetFile_ := AltFileName;
  end;


  erroneous_ := not downloadToFile(url_, targetPath_, targetFile_,
                                   proxy_, port_,
                                   'DownloadServiceThread ['+targetFile_+']> ', logger_);
  }
  done_ := true;
end;

end.
