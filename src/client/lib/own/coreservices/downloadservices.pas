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


type TDownloadWUJobServiceThread = class(TDownloadServiceThread)
   protected
    procedure Execute; override;
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

procedure TDownloadWUJobServiceThread.execute();
var AltFilename : String;
    index       : Longint;
begin
    logHeader_   := '[TDownloadWUJobServiceThread]> ';
    if not workflowman_.getClientJobQueueWorkflow().findRowInStatusForWURetrieval(jobqueuerow_) then
         begin
           logger_.log(LVL_DEBUG, logHeader_+'No jobs found in status FOR_WU_RETRIEVAL. Exit.');
           done_      := True;
           erroneous_ := false;
           Exit;
         end;

    if (Trim(jobqueuerow_.workunitjobpath)='') then
        begin
          logger_.log(LVL_SEVERE, logHeader_+'Internal error: found job in status FOR_WU_RETRIEVAL, but workunit is not defined.');
          workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Internal error: found job in status FOR_WU_RETRIEVAL, but workunit is not defined.');
          done_      := True;
          erroneous_ := True;
          Exit;
        end
    else
    begin
          // Here comes the main loop to retrieve a workunit
          workflowman_.getClientJobQueueWorkflow().changeStatusFromForWURetrievalToRetrievingWorkunit(jobqueuerow_);

          targetPath_ := ExtractFilePath(jobqueuerow_.workunitjobpath);
          targetFile_ := jobqueuerow_.workunitjob;
          url_ := srv_.url+'/workunits/jobs/'+jobqueuerow_.workunitjob;

          adaptFileNameIfItAlreadyExists;


          erroneous_ := not downloadToFile(url_, targetPath_, targetFile_,
                        proxy_, port_,
                        'DownloadServiceThread ['+targetFile_+']> ', logger_);

          if not (erroneous_) then
          begin
              workflowman_.getClientJobQueueWorkflow().changeStatusFromRetrievingWorkunitToWorkunitRetrieved(jobqueuerow_);
              if jobqueuerow_.requireack then
                  workflowman_.getClientJobQueueWorkflow().changeStatusFromWorkunitRetrievedToForAcknowledgement(jobqueuerow_)
              else
                  workflowman_.getClientJobQueueWorkflow().changeStatusFromWorkUnitRetrievedToReady(jobqueuerow_, logHeader_+'Fast transition: jobqueue does not require acknowledgement.');
          end
          else workflowman_.getClientJobQueueWorkflow().changeStatusToError(jobqueuerow_, logHeader_+'Communication problem: unable to retrieve workunit from server!');
    end;

  done_ := true;
end;

end.
