unit testspecialcommands;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  specialcommands, pluginmanagers, loggers, methodcontrollers,
  frontendmanagers, resultcollectors, stacks, stkconstants;

type

  TTestSpecialCommand= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestSpecialCommand;

  private
    plugman_  : TPluginManager;
    meth_     : TMethodController;
    logger_   : TLogger;
    res_      : TResultCollector;
    frontman_ : TFrontendManager;
    spec_     : TSpecialCommand;
  end; 

implementation

procedure TTestSpecialCommand.TestSpecialCommand;
var specArgType : TStkArgType;
    stk         : TStack;
    str         : TStkString;
begin
  AssertEquals('plugin.list is special command', true, spec_.isSpecialCommand('plugin.list', specArgType));
  AssertEquals('Argument type is', STK_ARG_SPECIAL_CALL_PLUGIN, specArgType);

  AssertEquals('core.version is special command', true, spec_.isSpecialCommand('core.version', specArgType));
  AssertEquals('Argument type is', STK_ARG_SPECIAL_CALL_CORE, specArgType);

  AssertEquals('plugin.asdfasf is not a special command', false, spec_.isSpecialCommand('plugin.asdfasf', specArgType));
  AssertEquals('Argument type is', STK_ARG_UNKNOWN, specArgType);

  clearStk(stk);
  AssertEquals('core.version', true, spec_.execCoreCommand('core.version',stk));
  popStr(str, stk);
  AssertEquals('Core version is', CORE_VERSION, str);
end;

procedure TTestSpecialCommand.SetUp; 
var path, extension : String;
    error           : TStkError;
begin
 clearError(error);
 path            := ExtractFilePath(ParamStr(0));
 extension       := 'dll';
 logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
 plugman_        := TPluginManager.Create(path+PathDelim+'plugins'+PathDelim+'lib', extension, logger_);
 plugman_.loadAll(error);
 meth_           := TMethodController.Create();
 res_            := TResultCollector.Create();
 frontman_       := TFrontendManager.Create();
 spec_           := TSpecialCommand.Create(plugman_, meth_, res_, frontman_);
end;

procedure TTestSpecialCommand.TearDown; 
begin
 spec_.Free;
 frontman_.Free;
 res_.Free;
 meth_.Free;
 plugman_.Free;
 logger_.Free;
end; 

initialization

  RegisterTest(TTestSpecialCommand); 
end.

