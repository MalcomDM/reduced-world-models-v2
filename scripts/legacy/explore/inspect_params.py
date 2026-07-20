from app.models.image_reconstructor import ImageReconstructor
from app.models.reduced_world_model import ReducedWorldModel

def main(model_name:str = "RWM"):
    # Instantiate your model (on CPU is fine for counting)
    if model_name=='RWM':
        model = ReducedWorldModel()
    elif model_name=='IR':
        model = ImageReconstructor()
    else:
        print("couldn't find this model")
        return None
    total_params = 0

    print(f"{'Parameter Name':50s} {'Shape':25s} {'# Params':>10s}")
    print("=" * 90)
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        print(f"{name:50s} {str(list(param.shape)):25s} {n:10,d}")

    print("=" * 90)
    print(f"{'Total':50s} {'':25s} {total_params:10,d}")


if __name__ == "__main__":
    main()