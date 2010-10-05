unit testcomputationthreads;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  computationthreads, pluginmanagers, methodcontrollers,
  loggers, resultcollectors, frontendmanagers, specialcommands,
  jobs, stacks;

type

  TTestComputationThread= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestComputationThread;
  private
    plugman_  : TPluginManager;
    meth_     : TMethodController;
    logger_   : TLogger;
    res_      : TResultCollector;
    frontman_ : TFrontendManager;
    spec_     : TSpecialCommand;
    job_      : TJob;

    procedure execJob(job : String);
  end; 

implementation

procedure TTestComputationThread.execJob(job : String);
var cthread : TComputationThread;
begin
 job_.Job := job;
 cthread := TJobParser.Create(plugman_, meth_, res_, frontman_, job_, 1);
 while (not cthread.isJobDone) do Sleep(100);
 cthread.Free;
end;

procedure TTestComputationThread.TestComputationThread;
begin
 execJob('1,1,add');
 Assert
end;

procedure TTestComputationThread.SetUp; 
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
end;

procedure TTestComputationThread.TearDown; 
begin
 spec_.Free;
 frontman_.Free;
 res_.Free;
 meth_.Free;
 plugman_.Free;
 logger_.Free;
 job_.Free;
end; 

initialization

  RegisterTest(TTestComputationThread); 
end.

