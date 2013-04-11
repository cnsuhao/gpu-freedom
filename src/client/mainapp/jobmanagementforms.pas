unit jobmanagementforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  ExtCtrls, ComCtrls, jobapis;

type

  { TJobManagementForm }

  TJobManagementForm = class(TForm)
    pnCreateJob: TPanel;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private

  public

  end;

var
  JobManagementForm: TJobManagementForm;

implementation

{ TJobManagementForm }

procedure TJobManagementForm.FormCreate(Sender: TObject);
begin
  //
end;

procedure TJobManagementForm.FormDestroy(Sender: TObject);
begin
  //
end;

initialization
  {$I jobmanagementforms.lrs}

end.

