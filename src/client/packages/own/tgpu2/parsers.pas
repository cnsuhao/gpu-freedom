unit parser;

interface

uses argretrievers, stacks, pluginmanager, methodcontrollers, specialcommands;

type TGPUParser = class(TObject);
 public 
   constructor Create(var plugman : TPluginManager; var meth : TMethodController; 
                      var specCommands : TSpecialCommand;
                      var job : TJob; threadId : Longint);
   destructor Destroy();
   
   function parse() : Boolean; overload;  
   function parse(jobStr : String; var stk : TStack; var error : TGPUError) : Boolean; overload;
   
 private
   plugman_        : TPluginManager;
   methController_ : TMethodController;
   speccommands_   : TSpecialCommand;  
   thrdId_         : Longint;
   job_            : TJob;
end;


implementation

constructor TGPUParser.Create(var plugman : TPluginManager; var meth : TMethodController; var specCommands : TSpecialCommand; var job : TJob; threadId : Longint);
begin
  inherited Create();
  plugMan_ := plugman;
  methController_ := meth;
  speccommands_ := specCommands;
  job_ := job;
  thrdId_ := threadId;
end;

destructor TGPUParser.Destroy();
begin
  inherited;
end;


procedure TGPUParser.parse() : Boolean;; overload;
begin
   Result := parse(job_.job, job_.Stack, job_.error);
   job_.hasError := (job._error.ErrorId>0);
end;

procedure TGPUParser.parse(jobStr : String; var stk : TStack; var error : TGPUError); overload;
var arg    : TGPUArg;
    argRetriever : TArgRetriever;
    isOK         : Boolean;
	pluginName   : String;
begin
  Result := False;
  argRetriever := TArgRetriever.Create(jobStr);
  hasErrors := false;
  
  while (argRetriever_.hasArguments() and (isOK)) do
     begin
       arg := getArgument(error);
       case arg.argType of
            GPU_ERROR :  isOK := false; // the error structure contains the error
            GPU_FLOAT : // a float was detected
                     isOK := loadExtendedOnStack(arg.argvalue, stk, error); 
                   
            GPU_STRING : // a string was detected
                     isOK := loadStringOnStack(arg.argstring, stk, error);
            GPU_BOOLEAN :
                     isOK := loadBooleanOnStack(arg.value, stk, error);
            GPU_EXPRESSION :
                     // we found an expression, we need to recursively call this method
                     isOK := parse(arg.argstring, stk, error);
            GPU_SPECIAL_CALL :
                     isOK := speccommands_.execSpecialCommand(arg.argstring, stk, error);            
            GPU_CALL :
                   begin
                     isOK := plugman_.method_exists(arg.argstring, pluginName, error);
					 if isOK then
					      begin
						    methController_.registerMethodCall(arg.argstring, pluginName, thrdID_);
							try
							  isOK := plugman_.method_execute(arg.argstring, pluginName, error);
							except (Exception e)
							  error.errorID := PLUGIN_THREW_EXCEPTION_ID;
							  error.errorMsg := PLUGIN_THREW_EXCEPTION;
							  error.errorArg := e.Message;
							  isOK := false;
							end;
							methController_.unregisterMethodCall(thrdID_);
						  end;

                   end;				   
       
       end; // case
     
     end; // while
   
   argRetriever.Free;
   Result := isOK;
end;

end.