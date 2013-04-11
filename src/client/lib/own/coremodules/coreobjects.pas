unit coreobjects;
{
 Defines a set of objects in common between mainapp (GUI) and core
 (c) 2011 by HB9TVM and the GPU project
}


interface

uses
  lockfiles, loggers, SysUtils,
  coreconfigurations, dbtablemanagers, servermanagers,
  coremodules, servicefactories, servicemanagers, compservicemanagers,
  workflowmanagers, identities, pluginmanagers, methodcontrollers,
  resultcollectors, frontendmanagers, downloadservicemanagers,
  uploadservicemanagers;

var
   logger         : TLogger;
   conf           : TCoreConfiguration;
   tableman       : TDbTableManager;
   serverman      : TServerManager;
   coremodule     : TCoreModule;
   servicefactory : TServiceFactory;
   serviceman     : TServiceThreadManager;
   compserviceman : TCompServiceThreadManager;
   workflowman    : TWorkflowManager;
   downserviceman : TDownloadServiceManager;
   upserviceman   : TUploadServiceManager;

   // lockfiles
   lf_morefrequentupdates : TLockFile;



procedure loadCommonObjects(logfile, componentname : String; corenumber : Longint);
procedure loadCoreObjects();
procedure discardCoreObjects;
procedure discardCommonObjects;

implementation

procedure loadCommonObjects(logfile, componentname : String; corenumber : Longint);
var
   path,
   logName : String;
begin
  path := extractFilePath(ParamStr(0));

  if corenumber=-1 then logName := logfile else logName := logfile+'_'+IntToStr(corenumber);

  logger    := TLogger.Create(path+PathDelim+'logs', logName+'.log', logName+'.old', LVL_DEBUG, 1024*1024);
  conf      := TCoreConfiguration.Create(path);
  conf.loadConfiguration();

  logger.setLogLevel(myConfId.loglevel);
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.logCR; logger.logCR;
  logger.log(LVL_INFO, '********************');
  if corenumber>=0 then
     logger.log(LVL_INFO, '* '+componentname+' launched (Core '+IntToStr(corenumber)+')')
  else
     logger.log(LVL_INFO, '* '+componentname+' launched ...');
  logger.log(LVL_INFO, '********************');


  tableman          := TDbTableManager.Create(path+PathDelim+'gpucore.db');
  tableman.OpenAll;
  serverman         := TServerManager.Create(conf, tableman.getServerTable(), logger);
  lf_morefrequentupdates := TLockFile.Create(path+PathDelim+'locks', 'morefrequentchat.lock');

  workflowman       := TWorkflowManager.Create(tableman, logger);
  coremodule        := TCoreModule.Create(logger, path, 'dll');
  servicefactory    := TServiceFactory.Create(workflowman, serverman, tableman, myConfId.proxy, myconfId.port, logger, conf, coremodule);
  serviceman        := TServiceThreadManager.Create(tmServiceStatus.maxthreads, logger);
end;

procedure loadCoreObjects();
var
   path : String;
begin
  path := extractFilePath(ParamStr(0));

  compserviceman    := TCompServiceThreadManager.Create(tmCompStatus.maxthreads, logger);
  downserviceman    := TDownloadServiceManager.Create(tmDownStatus.maxthreads, logger);
  upserviceman      := TUploadServiceManager.Create(tmUploadStatus.maxthreads, logger);
end;

procedure discardCoreObjects;
begin
  compserviceman.Free;
  downserviceman.Free;
  upserviceman.Free;
end;

procedure discardCommonObjects;
begin
  servicefactory.free;
  workflowman.free;
  coremodule.free;
  serviceman.free;
  serverman.Free;

  tableman.CloseAll;
  tableman.Free;
  conf.Free;
  logger.Free;

  //lf_morefrequentupdates.Free;
end;


end.
