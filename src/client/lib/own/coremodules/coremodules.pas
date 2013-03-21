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
  frontendmanagers, loggers, stkconstants;

type TCoreModule = class(TObject)
    constructor Create(logger : TLogger; path, extension : String);
    destructor Destroy;

    // helper structures
    function getPluginManager()       : TPluginManager;
    function getMethController()      : TMethodController;
    function getResultCollector()     : TResultCollector;
    function getFrontendManager()     : TFrontendManager;

    function getLogger()              : TLogger;
  private
    // core components
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    rescoll_        : TResultCollector;
    frontman_       : TFrontendManager;
    logger_         : TLogger;
end;

implementation

constructor TCoreModule.Create(logger : TLogger; path, extension : String);
begin
   inherited Create();
   logger_         := logger;
   plugman_        := TPluginManager.Create(path+PLUGIN_FOLDER+PathDelim+LIB_FOLDER, extension, logger_);
   methController_ := TMethodController.Create();
   rescoll_        := TResultCollector.Create();
   frontman_       := TFrontendManager.Create();
end;

destructor TCoreModule.Destroy;
begin
  methController_.Free;
  rescoll_.Free;
  frontman_.Free;
  plugman_.Free;
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

function TCoreModule.getLogger()          : TLogger;
begin
 Result := logger_;
end;



end.

