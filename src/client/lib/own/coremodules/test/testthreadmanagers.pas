unit testthreadmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  computationthreads, pluginmanagers, methodcontrollers,
  loggers, resultcollectors, frontendmanagers, specialcommands,
  jobs, stacks, compthreadmanagers;

type

  TTestThreadManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestThreadManager;
  private
    plugman_   : TPluginManager;
    meth_      : TMethodController;
    logger_    : TLogger;
    res_       : TResultCollector;
    frontman_  : TFrontendManager;
    spec_      : TSpecialCommand;
    job_       : TJob;
    threadman_ : TCompThreadManager;
  end; 

implementation

procedure TTestThreadManager.TestThreadManager;
begin


end;

procedure TTestThreadManager.SetUp; 
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
 job_            := TJob.Create();
 threadman_      := TCompThreadManager.Create(plugman_, meth_, res_, frontman_);
end;

procedure TTestThreadManager.TearDown; 
begin
 spec_.Free;
 frontman_.Free;
 res_.Free;
 meth_.Free;
 plugman_.Free;
 logger_.Free;
 job_.Free;
 threadman_.Free;
end; 

initialization

  RegisterTest(TTestThreadManager); 
end.

