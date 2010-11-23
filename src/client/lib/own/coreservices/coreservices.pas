unit coreservices;


interface

uses
 downloadthreadmanagers;

type TCoreService = class(TObject)
  public
    constructor Create(downMan : TDownloadThreadManager);

    function isEnabled : Boolean;
    procedure setEnabled(enabled : Boolean);

  protected
    enabled_ : Boolean;
    downMan_ : TDownloadThreadManager;
end;


type TReceiveService = class(TCoreService)
end;

type TTransmitService = class(TCoreService)
end;


implementation

constructor TCoreService.Create(downMan : TDownloadThreadManager);
begin
  inherited Create;
  enabled_ := true;
  downMan_ := downMan;
end;


function TCoreService.isEnabled : Boolean;
begin
 Result := enabled_;
end;

procedure TCoreService.setEnabled(enabled : Boolean);
begin
 enabled_ := enabled;
end;


end.
