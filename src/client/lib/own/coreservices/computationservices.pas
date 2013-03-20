unit computationservices;
{
    TComputationService is a database aware ComputationThread which executes a job stored
    in TBJOBQUEUE. It first tries to retrieve a jobqueue entry in status READY,
    sets jobqueue to RUNNING. When the job is computed, it persists an entry in TBJOBRESULT
    and sets the jobqueue to COMPUTED.

  (c) by 2002-2013 HB9TVM and the GPU Development Team
   This unit is released under GNU Public License (GPL)
}
interface

uses  Classes,
      jobs, methodcontrollers, pluginmanagers, resultcollectors, frontendmanagers,
      jobparsers, managedthreads, computationthreads;

type
  TComputationServiceThread = class(TComputationThread)
   public
      
    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager);

   protected
    procedure Execute; override;
    

   private

   end;

implementation

constructor TComputationServiceThread.Create(var plugman : TPluginManager; var meth : TMethodController;
                                             var res : TResultCollector; var frontman : TFrontendManager);
begin
  inherited Create(plugman, meth, res, frontman); // running
end;


procedure  TComputationServiceThread.Execute;
var parser : TJobParser;
begin
   {
  // They need to be retrieved from TBJOBQUEUE
  job_ := job;
  thrdId_ := thrdId;
  }

 parser := TJobParser.Create(plugman_, methController_, rescoll_, frontman_, job_, thrdId_);
 parser.parse();
 parser.Free;

 done_ := true;
 erroneous_ := job_.hasError;
end;


end.
