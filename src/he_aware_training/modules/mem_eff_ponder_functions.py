import torch
from typing import Tuple

@torch.jit.script
def goldschmidt_step(N_curr: torch.Tensor, D_curr: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D_curr = D_curr * F
    return N_curr * F, D_curr, 2.0 - D_curr

@torch.jit.script
def newton_step(x: torch.Tensor, curr_y: torch.Tensor) -> torch.Tensor:
    return 0.5 * curr_y * (3.0 - x * (curr_y ** 2))

@torch.jit.script
def newton_div_step(x: torch.Tensor, curr_y: torch.Tensor) -> torch.Tensor:
    return curr_y * (2.0 - x * curr_y)


class AnalyticMemEffPonderGoldschmidt(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, num, den, alpha, beta, mask, fixed_iters, learnable_iters, F):
        ctx.save_for_backward(num, den, alpha, beta, mask, F)
        ctx.fixed_iters = fixed_iters
        ctx.learnable_iters = learnable_iters

        N = num
        D = den
        if F is None:
            F = alpha - beta * D

        for _ in range(fixed_iters):
            N, D, F = goldschmidt_step(N, D, F)

        result = N * mask[:, 0]
        for i in range(learnable_iters):
            N, D, F = goldschmidt_step(N, D, F)
            result = result + N * mask[:, i+1]

        return result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        num, den, alpha, beta, mask, F = ctx.saved_tensors
        fixed_iters = ctx.fixed_iters
        learnable_iters = ctx.learnable_iters

        # Analytic Gradients (ignoring ponder iterations)
        grad_num_analytic = grad_output / den
        grad_den_analytic = -num / (den ** 2) * grad_output 

        # Accumulators
        grad_mask = torch.zeros_like(mask)

        # Initialize Step 0 states
        N = num
        D = den
        if F is None:
            F = alpha - beta * D

        # TODO check "setting N = 1" and thus not having to recompute the steps! 

        # --- Base Iterations ---
        for _ in range(fixed_iters):
            N, D, F = goldschmidt_step(N, D, F)

        grad_mask[:, 0] = (N * grad_output).sum(list(range(1, N.ndim))).reshape(grad_mask[:, 0].shape)

        # --- Ponder Iterations ---
        for i in range(learnable_iters):
            N, D, F = goldschmidt_step(N, D, F)

            # Mask gradient
            grad_mask[:, i+1] = (N * grad_output).sum(list(range(1, N.ndim))).reshape(grad_mask[:, i+1].shape)

        return grad_num_analytic, grad_den_analytic,  None, None, grad_mask, None, None, None
    

@torch.jit.script
def partial_goldschmidt_step(N: torch.Tensor, D: torch.Tensor, F: torch.Tensor, dN_dnum: torch.Tensor, dN_dden: torch.Tensor, dF_dden: torch.Tensor, first_iter: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    dN_dnum = dN_dnum * F
    dN_dden = dN_dden * F + N * dF_dden
    dF_dden = (first_iter * (1 + dF_dden) - 1) * F - D * dF_dden

    return dN_dnum, dN_dden, dF_dden, 1.0


class ExactMemEffPonderGoldschmidt(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, num, den, alpha, beta, mask, fixed_iters, learnable_iters, F):
        ctx.save_for_backward(num, den, alpha, beta, mask, F)
        ctx.fixed_iters = fixed_iters
        ctx.learnable_iters = learnable_iters

        N = num
        D = den
        if F is None:
            F = alpha - beta * D

        for _ in range(fixed_iters):
            N, D, F = goldschmidt_step(N, D, F)

        result = N * mask[:, 0]
        for i in range(learnable_iters):
            N, D, F = goldschmidt_step(N, D, F)
            result = result + N * mask[:, i+1]

        return result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        num, den, alpha, beta, mask, F = ctx.saved_tensors
        fixed_iters = ctx.fixed_iters
        learnable_iters = ctx.learnable_iters

        # Forward-mode AD: compute Jacobian-vector products alongside forward pass
        # Track derivatives w.r.t. each input: num, den, alpha, beta
        
        # Initialize primal values
        N = num
        D = den
        if F is None:
            F = alpha - beta * D
        
        dN_dnum = torch.ones_like(num)
        dN_dden = torch.zeros_like(den)
        dF_dden = -beta if beta is not None else torch.zeros_like(den)  # TODO: crashed when F was given

        first_iter = 0.0
        
        # Base iterations
        for _ in range(fixed_iters):
            dN_dnum, dN_dden, dF_dden, first_iter = partial_goldschmidt_step(N, D, F, dN_dnum, dN_dden, dF_dden, first_iter)
            # off-band denom rows amplify the JVP accumulators past bf16 (in-band max ~1e5)
            dN_dden = dN_dden.clamp(min=-1e6, max=1e6)
            dF_dden = dF_dden.clamp(min=-1e6, max=1e6)
            N, D, F = goldschmidt_step(N, D, F)

        # Initialize result derivatives
        dresult_dnum = torch.zeros_like(num)
        dresult_dden = torch.zeros_like(den)
        grad_mask = torch.zeros_like(mask)
        grad_mask[:, 0] = (N * grad_output).sum(list(range(1, N.ndim))).reshape(grad_mask[:, 0].shape)

        # Ponder iterations
        for i in range(learnable_iters):
            dN_dnum, dN_dden, dF_dden, first_iter = partial_goldschmidt_step(N, D, F, dN_dnum, dN_dden, dF_dden, first_iter)
            dN_dden = dN_dden.clamp(min=-1e6, max=1e6)
            dF_dden = dF_dden.clamp(min=-1e6, max=1e6)
            N, D, F = goldschmidt_step(N, D, F)

            # Accumulate into result
            dresult_dnum += dN_dnum * mask[:, i+1]
            dresult_dden += dN_dden * mask[:, i+1]

            # Mask gradient
            grad_mask[:, i+1] = (N * grad_output).sum(list(range(1, N.ndim))).reshape(grad_mask[:, i+1].shape)
        
        # Contract with grad_output to get final gradients
        grad_num = dresult_dnum * grad_output
        grad_den = dresult_dden * grad_output
        
        return grad_num, grad_den, None, None, grad_mask, None, None, None


class AnalyticMemEffPonderNewton(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, y_init, mask, iters):
        ctx.save_for_backward(x, y_init, mask)
        ctx.iters = iters

        curr_y = y_init

        # Base result
        result = curr_y * mask[:, 0]
        for i in range(iters):
            curr_y = newton_step(x, curr_y)
            result = result + curr_y * mask[:, i+1]

        return result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, y_init, mask = ctx.saved_tensors
        iters = ctx.iters

        # Analytic gradients, x is the x in 1/sqrt(x), y_init is the initial guess for 1/sqrt(x)
        grad_x_analytic = - 1 / (2 * x * torch.sqrt(x)) * grad_output
        grad_y = mask[:, 0] * grad_output

        # Mask gradients
        curr_y = y_init

        grad_mask = torch.zeros_like(mask)
        grad_mask[:, 0] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, 0].shape)

        for i in range(iters):
            curr_y = newton_step(x, curr_y)
            grad_mask[:, i+1] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, i+1].shape)

        return grad_x_analytic, grad_y, grad_mask, None

@torch.jit.script
def partial_newton_step(x: torch.Tensor, curr_y: torch.Tensor, dy_dx: torch.Tensor, dy_dy_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    partial_y = 1.5 - 1.5 * x * (curr_y ** 2)
    dy_dx = partial_y * dy_dx - 0.5 * (curr_y ** 3)
    dy_dy_init = partial_y * dy_dy_init

    return dy_dx, dy_dy_init
    

class ExactMemEffPonderNewton(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, y_init, mask, iters):
        ctx.save_for_backward(x, y_init, mask)
        ctx.iters = iters

        curr_y = y_init

        # Base result
        result = curr_y * mask[:, 0]
        for i in range(iters):
            curr_y = newton_step(x, curr_y)
            result = result + curr_y * mask[:, i+1]

        return result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, y_init, mask = ctx.saved_tensors
        iters = ctx.iters
        
        curr_y = y_init
        
        # Initialize tangent vectors
        dy_dx = torch.zeros_like(x)        # ∂y/∂x
        dy_dy_init = torch.ones_like(y_init)  # ∂y/∂y_init
        
        # Initialize result derivatives
        dresult_dx = torch.zeros_like(x)
        dresult_dy_init = torch.zeros_like(y_init)
        
        # Mask gradients
        grad_mask = torch.zeros_like(mask)
        grad_mask[:, 0] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, 0].shape)
        
        # Ponder iterations
        for i in range(iters):
            dy_dx, dy_dy_init = partial_newton_step(x, curr_y, dy_dx, dy_dy_init)
            curr_y = newton_step(x, curr_y)
            
            # Accumulate into result
            dresult_dx += dy_dx * mask[:, i+1]
            dresult_dy_init += dy_dy_init * mask[:, i+1]
            
            # Mask gradient
            grad_mask[:, i+1] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, i+1].shape)
        
        # Contract with grad_output
        grad_x = dresult_dx * grad_output
        # grad_y_init = (dresult_dy_init + mask[:, 0].reshape(dresult_dy_init.shape)) * grad_output
        grad_y_init = (dresult_dy_init + mask[:, 0]) * grad_output
        return grad_x, grad_y_init, grad_mask, None
    

class AnalyticMemEffPonderNewtonDiv(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, y_init, mask, iters):
        ctx.save_for_backward(x, y_init, mask)
        ctx.iters = iters

        curr_y = y_init

        # Base result
        result = curr_y * mask[:, 0]
        for i in range(iters):
            curr_y = newton_div_step(x, curr_y)
            result = result + curr_y * mask[:, i+1]

        return result

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        x, y_init, mask = ctx.saved_tensors
        iters = ctx.iters

        # Analytic gradients, x is the x in 1/x, y_init is the initial guess for 1/x
        grad_x_analytic = - 1 / (x ** 2) * grad_output
        grad_y = mask[:, 0] * grad_output

        # Mask gradients
        curr_y = y_init

        grad_mask = torch.zeros_like(mask)
        grad_mask[:, 0] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, 0].shape)

        for i in range(iters):
            curr_y = newton_div_step(x, curr_y)
            grad_mask[:, i+1] = (curr_y * grad_output).sum(list(range(1, curr_y.ndim))).reshape(grad_mask[:, i+1].shape)

        return grad_x_analytic, grad_y, grad_mask, None


if __name__ == "__main__":
    from he_aware_training.modules.learnable_components import logit_sequential_distribution
    from he_aware_training.approximation.classic import rational_remez, compute_optimal_linear_init, polyval_torch
    import numpy as np
    import math

    print("Testing AnalyticMemEffPonderGoldschmidt and ExactMemEffPonderGoldschmidt...")
    
    # Setup test inputs
    torch.manual_seed(42)
    batch_size = 2
    dim = 3
    fixed_iters = 3
    learnable_iters = 6
    halt_init_low = 0.05
    halt_init_high = 0.95
    halt_init_at = None

    halt_init_low = math.log(halt_init_low / (1 - halt_init_low))
    halt_init_high = math.log(halt_init_high / (1 - halt_init_high))

    init_logits = torch.ones(learnable_iters) * halt_init_low
    if halt_init_at is not None:
        init_logits[halt_init_at] = halt_init_high
    else:
        init_logits[-1] = halt_init_high
    
    mask = logit_sequential_distribution(init_logits)[0].to(torch.float64)  # Shape: (learnable_iters,)
    mask = mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)  # Shape: (batch_size, learnable_iters, 1)

    R_MIN = -64.0
    R_MAX = 0.0
    
    N_DEG = 4
    M_DEG = 4
    K_SQUARING = 4

    p_opt, q_opt, q_min, q_max = rational_remez(lambda x: np.exp(x), R_MIN / (2**K_SQUARING), R_MAX / (2**K_SQUARING), N_DEG, M_DEG)
    alpha, beta = compute_optimal_linear_init(q_min, q_max)
    alpha = torch.tensor(alpha, dtype=torch.float64)
    beta = torch.tensor(beta, dtype=torch.float64)

    xs = torch.linspace(R_MIN, R_MAX, batch_size * dim).reshape(batch_size, dim)
    num = polyval_torch(xs / (2**K_SQUARING), torch.from_numpy(p_opt).float()).to(torch.float64)
    den = polyval_torch(xs / (2**K_SQUARING), torch.from_numpy(q_opt).float()).to(torch.float64)
            
    print(f"\nInput shapes: num={num.shape}, den={den.shape}, mask={mask.shape}")
    print(f"Settings: fixed_iters={fixed_iters}, learnable_iters={learnable_iters}")
    
    # Test 1: Forward pass matches between both implementations
    print("\n" + "="*60)
    print("Test 1: Forward pass consistency")
    print("="*60)
    
    result_analytic = AnalyticMemEffPonderGoldschmidt.apply(
        num.clone(), den.clone(), alpha, beta, mask.clone(), fixed_iters, learnable_iters
    )
    result_exact = ExactMemEffPonderGoldschmidt.apply(
        num.clone(), den.clone(), alpha, beta, mask.clone(), fixed_iters, learnable_iters
    )
    actual_result = num/den  # Ground truth for comparison (ignoring ponder iterations)
    print(f"Analytic Result: {result_analytic}")
    print(f"Exact Result: {result_exact}")
    print(f"Ground Truth (num/den): {actual_result}")
    
    forward_match = torch.allclose(result_analytic, result_exact, rtol=1e-5, atol=1e-8)
    print(f"Forward outputs match: {forward_match}")
    if forward_match:
        print("✓ Both implementations produce identical forward results")
    else:
        print(f"✗ Forward difference: {(result_analytic - result_exact).abs().max().item()}")
    
    # Test 2: Gradient check for ExactMemEffPonderGoldschmidt
    print("\n" + "="*60)
    print("Test 2: Gradient correctness (ExactMemEffPonderGoldschmidt)")
    print("="*60)
    
    def exact_func(num, den, mask):
        return ExactMemEffPonderGoldschmidt.apply(
            num, den, alpha, beta, mask, fixed_iters, learnable_iters
        )
    
    num_test = num.clone().detach().requires_grad_(True)
    den_test = den.clone().detach().requires_grad_(True)
    mask_test = mask.clone().detach().requires_grad_(True)
    
    try:
        gradcheck_result = torch.autograd.gradcheck(
            exact_func, 
            (num_test, den_test, mask_test),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        if gradcheck_result:
            print("✓ ExactMemEffPonderGoldschmidt gradients are correct!")
        else:
            print("✗ ExactMemEffPonderGoldschmidt gradcheck failed")
    except Exception as e:
        print(f"✗ Gradcheck error: {e}")
    
    # Test 3: Compare gradients between implementations
    print("\n" + "="*60)
    print("Test 3: Gradient comparison (Analytic vs Exact)")
    print("="*60)
    
    num1 = num.clone().detach().requires_grad_(True)
    den1 = den.clone().detach().requires_grad_(True)
    mask1 = mask.clone().detach().requires_grad_(True)
    
    num2 = num.clone().detach().requires_grad_(True)
    den2 = den.clone().detach().requires_grad_(True)
    mask2 = mask.clone().detach().requires_grad_(True)
    
    out1 = AnalyticMemEffPonderGoldschmidt.apply(num1, den1, alpha, beta, mask1, fixed_iters, learnable_iters)
    out2 = ExactMemEffPonderGoldschmidt.apply(num2, den2, alpha, beta, mask2, fixed_iters, learnable_iters)
    
    loss1 = out1.sum()
    loss2 = out2.sum()
    
    loss1.backward()
    loss2.backward()
    
    grad_num_diff = (num1.grad - num2.grad).abs().max().item()
    grad_den_diff = (den1.grad - den2.grad).abs().max().item()
    grad_mask_diff = (mask1.grad - mask2.grad).abs().max().item()
    
    print(f"Max gradient difference for num: {grad_num_diff:.2e}")
    print(f"Max gradient difference for den: {grad_den_diff:.2e}")
    print(f"Max gradient difference for mask: {grad_mask_diff:.2e}")
    
    # Since analytic uses approximation, we expect some difference
    if grad_num_diff < 1e-3 and grad_den_diff < 1e-3:
        print("✓ Gradients are very close (Goldschmidt converged well)")
    elif grad_num_diff < 1e-1 and grad_den_diff < 1e-1:
        print("⚠ Gradients differ moderately (expected due to analytic approximation)")
    else:
        print("✗ Large gradient difference (Goldschmidt may not have converged)")
    
    # Test 4: Check that mask gradients match (both should be exact for this)
    print("\n" + "="*60)
    print("Test 4: Mask gradient consistency")
    print("="*60)
    
    mask_match = torch.allclose(mask1.grad, mask2.grad, rtol=1e-5, atol=1e-8)
    if mask_match:
        print("✓ Mask gradients match exactly between both implementations")
    else:
        print(f"✗ Mask gradient difference: {grad_mask_diff:.2e}")
    
    print("\n" + "="*60)
    print("GOLDSCHMIDT TESTS COMPLETED!")
    print("="*60)
    
    # ========================================================================
    # NEWTON TESTS
    # ========================================================================
    print("\n\n" + "="*60)
    print("Testing AnalyticMemEffPonderNewton and ExactMemEffPonderNewton...")
    print("="*60)
    
    # Setup test inputs for Newton (1/sqrt(x) approximation)
    torch.manual_seed(42)
    batch_size = 2
    dim = 3
    fixed_iters = 2
    learnable_iters = 8
    
    halt_init_low = 0.05
    halt_init_high = 0.95
    halt_init_at = None
    
    halt_init_low = math.log(halt_init_low / (1 - halt_init_low))
    halt_init_high = math.log(halt_init_high / (1 - halt_init_high))
    
    init_logits_newton = torch.ones(learnable_iters) * halt_init_low
    if halt_init_at is not None:
        init_logits_newton[halt_init_at] = halt_init_high
    else:
        init_logits_newton[-1] = halt_init_high
    
    mask_newton = logit_sequential_distribution(init_logits_newton)[0].to(torch.float64)
    mask_newton = mask_newton.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)  # Shape: (batch_size, learnable_iters, 1)
    
    print(f"Newton mask sum: {mask_newton.sum()}")
    
    # Create test data for 1/sqrt(x)
    x_vals = torch.linspace(0.5, 4.0, batch_size * dim).reshape(batch_size, dim).to(torch.float64)
    y_init = torch.randn_like(x_vals) * 0.1 + (1.0 / torch.sqrt(x_vals))  # Initial guess with noise
    
    print(f"\nInput shapes: x={x_vals.shape}, y_init={y_init.shape}, mask={mask_newton.shape}")
    print(f"Settings: fixed_iters={fixed_iters}, learnable_iters={learnable_iters}")
    
    # Test 1: Forward pass matches between both implementations
    print("\n" + "="*60)
    print("Test 1: Forward pass consistency (Newton)")
    print("="*60)
    
    result_analytic_newton = AnalyticMemEffPonderNewton.apply(
        x_vals.clone(), y_init.clone(), mask_newton.clone(), fixed_iters, learnable_iters
    )
    result_exact_newton = ExactMemEffPonderNewton.apply(
        x_vals.clone(), y_init.clone(), mask_newton.clone(), fixed_iters, learnable_iters
    )
    actual_result_newton = 1.0 / torch.sqrt(x_vals)  # Ground truth
    
    print(f"Analytic Result: {result_analytic_newton}")
    print(f"Exact Result: {result_exact_newton}")
    print(f"Ground Truth (1/sqrt(x)): {actual_result_newton}")
    
    forward_match_newton = torch.allclose(result_analytic_newton, result_exact_newton, rtol=1e-5, atol=1e-8)
    print(f"Forward outputs match: {forward_match_newton}")
    if forward_match_newton:
        print("✓ Both Newton implementations produce identical forward results")
    else:
        print(f"✗ Forward difference: {(result_analytic_newton - result_exact_newton).abs().max().item()}")
    
    # Test 2: Gradient check for ExactMemEffPonderNewton
    print("\n" + "="*60)
    print("Test 2: Gradient correctness (ExactMemEffPonderNewton)")
    print("="*60)
    
    def exact_newton_func(x, y_init, mask):
        return ExactMemEffPonderNewton.apply(x, y_init, mask, fixed_iters, learnable_iters)
    
    x_test = x_vals.clone().detach().requires_grad_(True)
    y_init_test = y_init.clone().detach().requires_grad_(True)
    mask_newton_test = mask_newton.clone().detach().requires_grad_(True)
    
    try:
        gradcheck_result_newton = torch.autograd.gradcheck(
            exact_newton_func,
            (x_test, y_init_test, mask_newton_test),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        if gradcheck_result_newton:
            print("✓ ExactMemEffPonderNewton gradients are correct!")
        else:
            print("✗ ExactMemEffPonderNewton gradcheck failed")
    except Exception as e:
        print(f"✗ Gradcheck error: {e}")
    
    # Test 3: Compare gradients between implementations
    print("\n" + "="*60)
    print("Test 3: Gradient comparison (Analytic vs Exact - Newton)")
    print("="*60)
    
    x1 = x_vals.clone().detach().requires_grad_(True)
    y_init1 = y_init.clone().detach().requires_grad_(True)
    mask_newton1 = mask_newton.clone().detach().requires_grad_(True)
    
    x2 = x_vals.clone().detach().requires_grad_(True)
    y_init2 = y_init.clone().detach().requires_grad_(True)
    mask_newton2 = mask_newton.clone().detach().requires_grad_(True)
    
    out_newton1 = AnalyticMemEffPonderNewton.apply(x1, y_init1, mask_newton1, fixed_iters, learnable_iters)
    out_newton2 = ExactMemEffPonderNewton.apply(x2, y_init2, mask_newton2, fixed_iters, learnable_iters)
    
    loss_newton1 = out_newton1.sum()
    loss_newton2 = out_newton2.sum()
    
    loss_newton1.backward()
    loss_newton2.backward()
    
    grad_x_diff = (x1.grad - x2.grad).abs().max().item()
    grad_y_init_diff = (y_init1.grad - y_init2.grad).abs().max().item()
    grad_mask_newton_diff = (mask_newton1.grad - mask_newton2.grad).abs().max().item()
    
    print(f"Max gradient difference for x: {grad_x_diff:.2e}")
    print(f"Max gradient difference for y_init: {grad_y_init_diff:.2e}")
    print(f"Max gradient difference for mask: {grad_mask_newton_diff:.2e}")
    
    # Analytic uses straight-through, so we expect differences in y_init and x gradients
    if grad_x_diff < 1e-3 and grad_y_init_diff < 0.1:
        print("✓ Gradients are reasonably close (Newton converged well)")
    elif grad_x_diff < 1e-1:
        print("⚠ Gradients differ moderately (expected due to analytic approximation)")
    else:
        print("✗ Large gradient difference (Newton may not have converged)")
    
    # Test 4: Check that mask gradients match
    print("\n" + "="*60)
    print("Test 4: Mask gradient consistency (Newton)")
    print("="*60)
    
    mask_match_newton = torch.allclose(mask_newton1.grad, mask_newton2.grad, rtol=1e-5, atol=1e-8)
    if mask_match_newton:
        print("✓ Mask gradients match exactly between both Newton implementations")
    else:
        print(f"✗ Mask gradient difference: {grad_mask_newton_diff:.2e}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)