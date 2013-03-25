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
  managedthreads, servermanagers, workflowmanagers, dbtablemanagers, jobqueuetables,
  downloadutils, loggers, sysutils;


type TDownloadServiceThread = class(TManagedThread)
 public
   constructor Create(var srv : TServerRecord; var tableman : TDbTableManager; var workflowman : TWorkflowManager;
                      proxy, port : String; var logger : TLogger);

 protected
    procedure Execute; override;

 private
    url_,
    proxy_,
    port_,
    logHeader_,
    targetPath_,
    targetFile_  : String;
    srv_         : TServerRecord;
    tableman_    : TDbTableManager;
    workflowman_ : TWorkflowManager;
    logger_      : TLogger;

    jobqueuerow_ : TDbJobQueueRow;

    procedure adaptFileNameIfItAlreadyExists;
end;


implementation


constructor TDownloadServiceThread.Create(var srv : TServerRecord; var tableman : TDbTableManager; var workflowman : TWorkflowManager;
                                   proxy, port : String; var logger : TLogger);
begin
  inherited Create(true); // suspended

  srv_         := srv;
  tableman_    := tableman;
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

    if (Trim(jobqueuerow_.workunitjobpath)='') and (not jobqueuerow_.islocal) then
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

          targetPath_ := ExtractFilePath(jobqueuerow_.workunitjobpath);
          targetFile_ := jobqueuerow_.workunitjob;
          url_ := srv_.url+'/workunits/jobs/'+jobqueuerow_.workunitjob;

          adaptFileNameIfItAlreadyExists;


          erroneous_ := not downloadToFile(url_, targetPath_, targetFile_,
                        proxy_, port_,
                        'DownloadServiceThread ['+targetFile_+']> ', logger_);

          if not (erroneous_) then
          begin
              workflowman_.getJobQueueWorkflow().changeStatusFromRetrievingWorkunitToWorkunitRetrieved(jobqueuerow_);
              if not jobqueuerow_.requireack then
                  workflowman_.getJobQueueWorkflow().changeStatusFromWorkUnitRetrievedToReady(jobqueuerow_, logHeader_+'Fast transition: jobqueue does not require acknowledgement.');
          end
          else workflowman_.getJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Unable to retrieve workunit!');
    end;

  done_ := true;
end;


procedure TDownloadServiceThread.adaptFileNameIfItAlreadyExists;
var index : Longint;
    AltFileName : String;
begin
    if FileExists(targetPath_+targetFile_) then
    begin
      index := 2;
      repeat
        AltFileName := targetFile_ + '.' + IntToStr(index);
        inc(index);
      until not FileExists(targetPath_+AltFileName);
      targetFile_ := AltFileName;

      logger_.log(LVL_WARNING, logHeader_+'"'+targetFile_+'" exists, writing to "'+targetFile_+'"');
      jobqueuerow_.workunitjob     := targetFile_;
      jobqueuerow_.workunitjobpath := targetPath_+targetFile_;
      tableman_.getJobQueueTable().insertOrUpdate(jobqueuerow_);
    end;

end;

end.
