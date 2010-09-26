unit coremodules;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils,
  pluginmanagers, methodcontrollers, specialcommands, resultcollectors,
  frontendmanagers;

implementation

type TCoreModules = class(TObject)
    constructor Create(var plugman : TPluginManager);
    destructor Destroy;

    // helper structures
    function getPluginManager()    : TPluginManager;
    function getMethController()  : TMethodController;
    function getSpecCommands()    : TSpecialCommand;
    function getResultCollector() : TResultCollector;
    function getFrontendManager() : TFrontendManager;

    // core components
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    speccommands_   : TSpecialCommand;
    rescoll_        : TResultCollector;
    frontman_       : TFrontendManager;
end;


constructor TCoreModules.Create(var plugman : TPluginManager);
begin
   inherited;
   plugman_ := plugman;
   methController_ := TMethodController.Create();
   rescoll_        := TResultCollector.Create();
   frontman_       := TFrontendManager.Create();
   speccommands_   := TSpecialCommands.Create(plugman_, methController_, rescoll_, frontman_);
end;

destructor TCoreModules.Destroy;
begin
  speccommands_.Free;
  methController_.Free;
  rescoll_.Free;
  frontman_.Free;
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


end;

end.

