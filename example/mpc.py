from invariance import Control
from casadi import MX, Function, Opti

class MPCController (Control) :
    def __init__(self, Q, R, mode='hybrid'):
        super().__init__(mode)

        x = MX.sym('x',4,1)
        u = MX.sym('u',2,1)
        px = x[0]; vx = x[1]; py = x[2]; vy = x[3]
        ux = u[0]; uy = u[1]

        xdot = MX(4,1)
        xdot[0] = vx
        xdot[1] = ux
        xdot[2] = vy
        xdot[3] = uy

        self.N = 40
        step = 0.125

        f = Function('f',[x,u],[xdot],['x','u'],['xdot'])

        self.opti = Opti()
        self.xx = self.opti.variable(4,self.N+1)
        self.uu = self.opti.variable(2,self.N)
        self.x0 = self.opti.parameter(4,1)
        self.slack = self.opti.variable(1,self.N)

        self.opti.subject_to(self.xx[:,0] == self.x0)
        J = 0
        for n in range(self.N) :
            self.opti.subject_to(self.xx[:,n+1] == self.xx[:,n] + step*f(self.xx[:,n], self.uu[:,n]))
            # J += self.xx[0,n]**2 + self.xx[1,n]**2 + self.xx[2,n]**2 + self.xx[3,n]**2
            # J += 0.5*self.uu[0,n]**2 + 0.5*self.uu[1,n]**2
            J += self.xx[:,n].T @ Q @ self.xx[:,n]
            J += self.uu[:,n].T @ R @ self.uu[:,n]
            J += 1e6*self.slack[0,n]**2
            self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]+4)**2 + (self.xx[1,n]-4)**2 >= 3**2 - self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]-4)**2 + (self.xx[1,n]+4)**2 >= 3**2 - self.slack[0,n])
            self.opti.subject_to((self.xx[0,n]+4)**2 + (self.xx[1,n]+4)**2 >= 3**2 - self.slack[0,n])

        self.opti.minimize(J)
        self.opti.subject_to(self.opti.bounded(-20,self.xx[0,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[1,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[2,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.xx[3,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.uu[0,:],20))
        self.opti.subject_to(self.opti.bounded(-20,self.uu[1,:],20))

        self.opti.solver('ipopt',{'print_time':0},{'linear_solver':'MA57', 'print_level':0, 'sb':'yes','max_iter':100000})
        # self.opti.solver('ipopt',{},{'linear_solver':'MA57', 'sb':'yes','max_iter':100000})
        # self.opti.solver('ipopt',{'print_time':0},{'print_level':0, 'sb':'yes','max_iter':100000})

    def u (self, t, x) :
        self.opti.set_value(self.x0, x)
        for n in range(self.N + 1) :
            self.opti.set_initial(self.xx[:,n], x)
        sol = self.opti.solve()
        # print(sol.value(self.slack))
        return sol.value(self.uu[:,0])
