unit jobmanagementforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, sqlite3conn, FileUtil, LResources, Forms, Controls,
  Graphics, Dialogs, ExtCtrls, ComCtrls, StdCtrls, Spin, jobapis, coreobjects;

type

  { TJobManagementForm }

  TJobManagementForm = class(TForm)
    btnSelectInputWorkunit: TButton;
    btnSubmitJob: TButton;
    cbJobBox: TComboBox;
    cbJobType: TComboBox;
    cbRequireAck: TCheckBox;
    cbTagOutputWorkunit: TCheckBox;
    cgWorkflow: TCheckGroup;
    cbTagInputWorkunit: TCheckBox;
    edtWorkunitInput: TEdit;
    edtWorkunitOutput: TEdit;
    gbWorkunits: TGroupBox;
    lblNumberOfRequests: TLabel;
    lblOutputWorkunitName: TLabel;
    lblWorkunitInput: TLabel;
    lblJobType: TLabel;
    lblJob: TLabel;
    lblJobDefinitionIdDesc: TLabel;
    lblJobDefinitionId: TLabel;
    OpenDialog: TOpenDialog;
    pnCreateJob: TPanel;
    rbGlobal: TRadioButton;
    rbLocal: TRadioButton;
    seNbRequests: TSpinEdit;
    SQLite3Connection1: TSQLite3Connection;
    procedure btnSubmitJobClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure SQLite3Connection1AfterConnect(Sender: TObject);

  private

  public

  end;

var
  JobManagementForm: TJobManagementForm;

implementation

{ TJobManagementForm }

procedure TJobManagementForm.FormCreate(Sender: TObject);
begin
  Visible := True;
end;

procedure TJobManagementForm.btnSubmitJobClick(Sender: TObject);
var job : TGPUJobApi;
begin
  job.job        := cbJobBox.Text;
  job.jobtype    := 'GPU_Engine';
  job.islocal    := rbLocal.Checked;
  job.requireack := cbRequireAck.Checked;
  job.trandetails.nbrequests     := seNbRequests.Value;
  job.trandetails.workunitjob    := Trim(edtWorkunitInput.Text);
  job.trandetails.tagwujob       := cbTagInputWorkunit.Checked;
  job.trandetails.workunitresult := Trim(edtWorkunitOutput.Text);
  job.trandetails.tagwuresult    := cbTagOutputWorkunit.Checked;

  jobapi.createJob(job);
  lblJobDefinitionId.Caption := job.jobdefinitionid;

end;

procedure TJobManagementForm.FormDestroy(Sender: TObject);
begin
  //
end;

procedure TJobManagementForm.SQLite3Connection1AfterConnect(Sender: TObject);
begin

end;


initialization
  {$I jobmanagementforms.lrs}

end.

