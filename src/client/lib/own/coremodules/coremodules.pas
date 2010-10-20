unit coremodules;
{
  Unit CoreModules is the entry point for this GPU component.
  It instantiates most of the structures which compose the core
  of GPU. To instantiate it, it is necessary to pass the path and the extension
  of plugins.

  ThreadManager instantiates ComputationThreads which
  effectively perform the computation.

  PluginManager administrates plugins.

  FrontendManager administrates the queue of registered jobs for frontends.

  MethController checks that the same function in plugins is not called
  concurrently for increased stability of the whole core.

  ResultCollector collects results and computes stuff like average and
  standard deviation.

  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)

}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils,
  pluginmanagers, methodcontrollers, specialcommands, resultcollectors,
  frontendmanagers, compthreadmanagers, downloadthreadmanagers, loggers;

implementation

type TCoreModule = class(TObject)
    constructor Create(path, extension : String);
    destructor Destroy;

    // helper structures
    function getPluginManager()     : TPluginManager;
    function getMethController()    : TMethodController;
    function getResultCollector()   : TResultCollector;
    function getFrontendManager()   : TFrontendManager;
    function getCompThreadManager() : TCompThreadManager;
    function getDownThreadManager() : TDownloadThreadManager;
    function getLogger()            : TLogger;
  private
    // core components
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    rescoll_        : TResultCollector;
    frontman_       : TFrontendManager;
    compthreadman_  : TCompThreadManager;
    logger_         : TLogger;
    downthreadman_  : TDownloadThreadManager;
end;


constructor TCoreModule.Create(path, extension : String);
begin
   inherited Create();
   logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log', 'core.old', LVL_DEFAULT, 1024*1024);
   plugman_        := TPluginManager.Create(path+PathDelim+'plugins'+PathDelim+'lib', extension, logger_);
   methController_ := TMethodController.Create();
   rescoll_        := TResultCollector.Create();
   frontman_       := TFrontendManager.Create();
   compthreadman_  := TCompThreadManager.Create(plugman_, methController_, rescoll_, frontman_);
   downthreadman_  := TDownloadThreadManager.Create(logger_);
end;

destructor TCoreModule.Destroy;
begin
  methController_.Free;
  rescoll_.Free;
  frontman_.Free;
  compthreadman_.Free;
  downthreadman_.Free;
  logger_.Free;
  inherited;
end;

function TCoreModule.getMethController() : TMethodController;
begin
 Result := methController_;
end;


function TCoreModule.getResultCollector(): TResultCollector;
begin
 Result := rescoll_;
end;

function TCoreModule.getFrontendManager() : TFrontendManager;
begin
 Result := frontman_;
end;

function TCoreModule.getPluginManager()   : TPluginManager;
begin
 Result := plugman_;
end;

function TCoreModule.getCompThreadManager()   : TCompThreadManager;
begin
 Result := compthreadman_;
end;

function TCoreModule.getLogger()          : TLogger;
begin
 Result := logger_;
end;

function TCoreModule.getDownThreadManager() : TDownloadThreadManager;
begin
 Result := downthreadman_;
end;

end.

