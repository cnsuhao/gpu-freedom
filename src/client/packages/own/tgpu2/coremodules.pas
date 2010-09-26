unit coremodules;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils,
  pluginmanagers, methodcontrollers, specialcommands, resultcollectors,
  frontendmanagers, threadmanagers;

implementation

type TCoreModules = class(TObject)
    constructor Create(path, extension : String);
    destructor Destroy;

    // helper structures
    function getPluginManager()   : TPluginManager;
    function getMethController()  : TMethodController;
    function getSpecCommands()    : TSpecialCommand;
    function getResultCollector() : TResultCollector;
    function getFrontendManager() : TFrontendManager;
    function getThreadManager()   : TThreadManager;

    // core components
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    speccommands_   : TSpecialCommand;
    rescoll_        : TResultCollector;
    frontman_       : TFrontendManager;
    threadman_      : TThreadManager;
end;


constructor TCoreModules.Create(path, extension : String);
begin
   inherited;
   plugman_        := TPluginManager.Create(path, extension);
   methController_ := TMethodController.Create();
   rescoll_        := TResultCollector.Create();
   frontman_       := TFrontendManager.Create();
   threadman_      := TThreadManager.Create();
   speccommands_   := TSpecialCommands.Create(plugman_, methController_, rescoll_, frontman_, threadman_);
end;

destructor TCoreModules.Destroy;
begin
  speccommands_.Free;
  methController_.Free;
  rescoll_.Free;
  frontman_.Free;
  threadman_.Free;
  inherited;
end;

function TCoreModules.getMethController() : TMethodController;
begin
 Result := methController_;
end;

function TCoreModules.getSpecCommands() : TSpecialCommand;
begin
 Result := speccommands_;
end;

function TCoreModules.getResultCollector(): TResultCollector;
begin
 Result := rescoll_;
end;

function TCoreModules.getFrontendManager() : TFrontendManager;
begin
 Result := frontman_;
end;

function TCoreModules.getPluginManager()   : TPluginManager;
begin
 Result := plugman_;
end;

function getThreadManager()   : TThreadManager;
begin
 Result := threadman_;
end;

end;

end.

